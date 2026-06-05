"""
k8s.py — ContainerPathK8s

Submits a real Kubernetes Job to a Minikube cluster, waits for it to complete,
reads results from the shared volume, and returns the same 3-tuple interface
as ContainerPathSimulator so main.py needs no changes.

Prerequisites (run once — see setup instructions in README or implementation plan):
  1. minikube start --driver=docker --memory=4096 --cpus=2
  2. minikube mount "<project_root>\\storage:/mnt/cs450"  (keep terminal open)
  3. minikube docker-env | Invoke-Expression
     docker build -t video-analytics-worker:latest ./container
  4. kubectl create secret generic gemini-secret \\
         --from-literal=api_key=$env:GEMINI_API_KEY
  5. pip install kubernetes
"""

import os
import json
import shutil
import time
import uuid
import subprocess

# Cost model: t3.medium EC2 on-demand rate
_EC2_RATE_PER_SEC    = 0.0000115   # USD/sec
_S3_PUT_COST         = 0.000005    # USD per PUT request
_S3_GET_COST         = 0.0000004   # USD per GET request
_JOB_TIMEOUT_SEC     = 600         # 10 minutes max


class ContainerPathK8s:
    """
    Runs the video analytics pipeline as a real Kubernetes Job on Minikube.

    Shared storage convention (requires `minikube mount`):
      Host path (Windows)          ->  In-cluster path
      storage/k8s_input/{job_id}/  ->  /mnt/cs450/k8s_input/{job_id}/
      storage/k8s_results/{job_id}/->  /mnt/cs450/k8s_results/{job_id}/

    The minikube mount command maps the project's storage/ directory to
    /mnt/cs450 inside the Minikube VM, giving pods hostPath access.
    """

    def __init__(self, storage,
                 namespace:   str = "default",
                 image:       str = "video-analytics-worker:latest",
                 mount_host:  str = None,   # e.g. C:\\Users\\Seden\\Desktop\\CS450\\storage
                 mount_guest: str = "/mnt/cs450"):
        self.storage     = storage
        self.namespace   = namespace
        self.image       = image
        self.mount_guest = mount_guest.rstrip("/")

        # Default mount_host = <project_root>/storage
        if mount_host is None:
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mount_host = os.path.join(here, "storage")
        self.mount_host = mount_host

        self._k8s_ready = False
        self._batch_v1  = None
        self._core_v1   = None
        self._load_kube_config()

    # ------------------------------------------------------------------ #
    # Public API (same signature as ContainerPathSimulator)
    # ------------------------------------------------------------------ #

    def run(self, video_local_path, query, target_labels):
        """
        Submit a Kubernetes Job, wait for completion, return results.

        Returns:
            (output_s3_key | None, logs: list[str], annotation_data: dict)
        """
        logs    = []
        job_id  = str(uuid.uuid4())[:8]
        fname   = os.path.basename(video_local_path)

        def log(msg):
            full = f"[K8s-Worker] {msg}"
            print(full)
            logs.append(msg)

        if not self._k8s_ready:
            log("ERROR: Kubernetes client not available. Is Minikube running?")
            return None, logs, {}

        # ---------------------------------------------------------- #
        # 1. Copy video to the shared storage directory
        # ---------------------------------------------------------- #
        input_dir_host = os.path.join(self.mount_host, "k8s_input", job_id)
        os.makedirs(input_dir_host, exist_ok=True)
        dest_video = os.path.join(input_dir_host, fname)
        shutil.copy2(video_local_path, dest_video)
        log(f"Video copied to shared volume: {dest_video}")

        # Count one S3 PUT for cost tracking (no second file copy needed)
        self.storage.put_requests   += 1
        self.storage.bytes_uploaded += os.path.getsize(dest_video)

        # ---------------------------------------------------------- #
        # 2. Prepare the results directory
        # ---------------------------------------------------------- #
        results_dir_host  = os.path.join(self.mount_host, "k8s_results", job_id)
        os.makedirs(results_dir_host, exist_ok=True)

        # In-cluster paths (via the minikube mount)
        video_path_guest   = f"{self.mount_guest}/k8s_input/{job_id}/{fname}"
        output_dir_guest   = f"{self.mount_guest}/k8s_results/{job_id}"

        # ---------------------------------------------------------- #
        # 3. Build and submit the Kubernetes Job
        # ---------------------------------------------------------- #
        job_name = f"video-analytics-{job_id}"
        log(f"Submitting Kubernetes Job: {job_name}")

        job_body = self._build_job(
            job_name       = job_name,
            job_id         = job_id,
            query          = query,
            target_labels  = target_labels,
            video_path     = video_path_guest,
            output_dir     = output_dir_guest,
            input_dir_host = input_dir_host,
            results_dir_host = results_dir_host,
        )

        start_time = time.time()
        try:
            self._batch_v1.create_namespaced_job(
                namespace = self.namespace,
                body      = job_body,
            )
        except Exception as e:
            log(f"ERROR creating Job: {e}")
            return None, logs, {}

        # ---------------------------------------------------------- #
        # 4. Wait for Job completion
        # ---------------------------------------------------------- #
        log("Waiting for Job to complete...")
        succeeded, error_msg = self._wait_for_job(job_name, timeout=_JOB_TIMEOUT_SEC)
        elapsed = time.time() - start_time

        if not succeeded:
            log(f"Job failed or timed out: {error_msg}")
            self._print_pod_logs(job_name)     # ← show crash reason before cleanup
            self._delete_job(job_name)
            return None, logs, {}

        log(f"Job completed in {elapsed:.1f}s.")

        # ---------------------------------------------------------- #
        # 5. Read results from shared volume
        # ---------------------------------------------------------- #
        output_json_path = os.path.join(results_dir_host, "output.json")
        if not os.path.exists(output_json_path):
            log("ERROR: output.json not found — worker may have crashed.")
            self._delete_job(job_name)
            return None, logs, {}

        with open(output_json_path, encoding="utf-8") as f:
            worker_output = json.load(f)

        worker_logs = worker_output.get("logs", [])
        for wl in worker_logs:
            logs.append(f"[pod] {wl}")

        intervals         = [tuple(iv) for iv in worker_output.get("intervals", [])]
        frame_annotations = {
            float(k): v
            for k, v in worker_output.get("frame_annotations", {}).items()
        }
        annotated_video   = worker_output.get("annotated_video")

        # ---------------------------------------------------------- #
        # 6. Register output as S3 equivalent for cost accounting
        # ---------------------------------------------------------- #
        final_s3_key = None
        if annotated_video and os.path.exists(annotated_video):
            final_s3_key = f"outputs/{job_id}_annotated.mp4"
            self.storage.put_file(annotated_video, final_s3_key)
            log(f"Annotated video registered: {final_s3_key}")

        # ---------------------------------------------------------- #
        # 7. Compute cost (EC2 equivalent pricing on real elapsed time)
        # ---------------------------------------------------------- #
        compute_cost = elapsed * _EC2_RATE_PER_SEC
        s3_cost      = _S3_PUT_COST + _S3_GET_COST        # 1 PUT + 1 GET
        total_cost   = compute_cost + s3_cost
        log(f"Cost: ${total_cost:.6f} USD  "
            f"(compute {elapsed:.1f}s × ${_EC2_RATE_PER_SEC}/s + S3)")

        # ---------------------------------------------------------- #
        # 8. Cleanup Kubernetes Job
        # ---------------------------------------------------------- #
        self._delete_job(job_name)

        annotation_data = {
            "intervals":         intervals,
            "frame_annotations": frame_annotations,
        }
        return final_s3_key, logs, annotation_data

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_kube_config(self):
        """Load kubeconfig (written by `minikube start`)."""
        try:
            from kubernetes import client, config
            config.load_kube_config()
            self._batch_v1 = client.BatchV1Api()
            self._core_v1  = client.CoreV1Api()
            self._k8s_ready = True
        except Exception as e:
            print(f"[ContainerPathK8s WARNING] Could not load kubeconfig: {e}")
            print("  Make sure Minikube is running: minikube start")
            self._k8s_ready = False

    def _build_job(self, job_name, job_id, query, target_labels,
                   video_path, output_dir,
                   input_dir_host, results_dir_host):
        """Builds a kubernetes.client.V1Job object programmatically."""
        from kubernetes import client as k8s

        env_vars = [
            k8s.V1EnvVar(name="VIDEO_PATH",     value=video_path),
            k8s.V1EnvVar(name="QUERY",          value=query),
            k8s.V1EnvVar(name="TARGET_LABELS",  value=",".join(target_labels)),
            k8s.V1EnvVar(name="JOB_ID",         value=job_id),
            k8s.V1EnvVar(name="OUTPUT_DIR",     value=output_dir),
            # Gemini API key from Kubernetes Secret
            k8s.V1EnvVar(
                name="GEMINI_API_KEY",
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name="gemini-secret",
                        key="api_key",
                        optional=True,          # don't crash if secret missing
                    )
                )
            ),
        ]

        # Mount the entire /mnt/cs450 root (established at minikube start) so that
        # newly created per-job subdirectories are visible immediately — avoids the
        # race condition where Docker snapshots an empty per-job dir at pod start.
        volume_mounts = [
            k8s.V1VolumeMount(name="storage-vol", mount_path=self.mount_guest),
        ]

        volumes = [
            k8s.V1Volume(
                name="storage-vol",
                host_path=k8s.V1HostPathVolumeSource(
                    path=self.mount_guest   # /mnt/cs450 — already live via minikube mount
                )
            ),
        ]


        container = k8s.V1Container(
            name             = "worker",
            image            = self.image,
            image_pull_policy= "Never",         # use locally built image
            env              = env_vars,
            volume_mounts    = volume_mounts,
            resources        = k8s.V1ResourceRequirements(
                requests={"cpu": "500m", "memory": "1Gi"},
                limits  ={"cpu": "2",    "memory": "4Gi"},
            ),
        )

        pod_spec = k8s.V1PodSpec(
            restart_policy = "Never",
            containers     = [container],
            volumes        = volumes,
        )

        job_spec = k8s.V1JobSpec(
            template               = k8s.V1PodTemplateSpec(
                metadata = k8s.V1ObjectMeta(labels={"app": "video-analytics"}),
                spec     = pod_spec,
            ),
            backoff_limit          = 0,       # don't retry on failure
            ttl_seconds_after_finished = 300, # auto-delete after 5 min
        )

        return k8s.V1Job(
            api_version = "batch/v1",
            kind        = "Job",
            metadata    = k8s.V1ObjectMeta(name=job_name, namespace=self.namespace),
            spec        = job_spec,
        )

    def _wait_for_job(self, job_name: str, timeout: int = 600):
        """Polls until the Job succeeds or fails. Returns (succeeded, error_msg)."""
        from kubernetes.client.rest import ApiException
        deadline = time.time() + timeout
        poll_interval = 5  # seconds

        while time.time() < deadline:
            try:
                job = self._batch_v1.read_namespaced_job(
                    name=job_name, namespace=self.namespace
                )
                status = job.status
                if status.succeeded and status.succeeded > 0:
                    return True, None
                if status.failed and status.failed > 0:
                    return False, f"Job failed ({status.failed} failure(s))"
            except ApiException as e:
                return False, f"API error while polling: {e}"
            time.sleep(poll_interval)

        return False, f"Job timed out after {timeout}s"

    def _get_pod_logs(self, job_name: str) -> str:
        """Fetch logs from the most recent pod of the job (best-effort)."""
        try:
            pods = self._core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"job-name={job_name}"
            )
            if not pods.items:
                return "(no pods found)"
            pod = pods.items[-1]
            return self._core_v1.read_namespaced_pod_log(
                name=pod.metadata.name,
                namespace=self.namespace
            )
        except Exception as e:
            return f"(could not fetch logs: {e})"

    def _print_pod_logs(self, job_name: str):
        """Print pod logs to stdout so the failure reason is visible."""
        logs = self._get_pod_logs(job_name)
        print(f"\n{'─'*60}")
        print(f"[K8s pod logs for {job_name}]")
        print(f"{'─'*60}")
        print(logs or "(empty)")
        print(f"{'─'*60}\n")

    def _delete_job(self, job_name: str):
        """Delete the Job and its pods (cleanup)."""
        from kubernetes import client as k8s
        try:
            self._batch_v1.delete_namespaced_job(
                name            = job_name,
                namespace       = self.namespace,
                body            = k8s.V1DeleteOptions(propagation_policy="Foreground"),
            )
        except Exception:
            pass   # best-effort cleanup
