# container/__init__.py
# - simulator.py : ContainerPathSimulator (fast local fallback, no K8s needed)
# - k8s.py       : ContainerPathK8s       (real Kubernetes Job on Minikube)
# - worker.py    : entry point run inside the Kubernetes pod
# - Dockerfile   : builds the worker image

from container.k8s import ContainerPathK8s
from container.simulator import ContainerPathSimulator

__all__ = ["ContainerPathK8s", "ContainerPathSimulator"]
