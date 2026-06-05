# run.ps1 -- One-click setup and batch run for CS450 Video Analytics
#
# Usage (from project root in PowerShell):
#   .\run.ps1              -- simulator container path, no Minikube needed
#   .\run.ps1 -UseK8s      -- real Kubernetes via Minikube
#   .\run.ps1 -UseK8s -workload my_videos.csv

param(
    [switch]$UseK8s,
    [string]$workload = "workload.csv",
    [string]$model    = "llama3.2:1b"
)


$ErrorActionPreference = "Stop"
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $PROJECT_ROOT

function Info($msg)  { Write-Host "[INFO]  $msg" -ForegroundColor Cyan  }
function Ok($msg)    { Write-Host "[OK]    $msg" -ForegroundColor Green }
function Warn($msg)  { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Fail($msg)  { Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------------------
# Load .env file so GEMINI_API_KEY is available for kubectl secret creation
# ---------------------------------------------------------------------------
$envFile = Join-Path $PROJECT_ROOT ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#=][^=]*)=(.+)$') {
            [System.Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim())
        }
    }
    Info ".env loaded."
}

if (-not $env:GEMINI_API_KEY) {
    Warn "GEMINI_API_KEY not set. VLM calls will use fallback simulation."
}

# ===========================================================================
# Minikube setup (only when --k8s is passed)
# ===========================================================================
if ($UseK8s) {
    Info "K8s mode enabled. Configuring Minikube..."
    $STORAGE_PATH = Join-Path $PROJECT_ROOT "storage"

    # 1. Check Docker Desktop
    try {
        docker info 2>&1 | Out-Null
    } catch {
        Fail "Docker Desktop is not running. Start it and re-run this script."
    }
    Ok "Docker Desktop is running."

    # 2. Start Minikube if not running
    $mkStatus = ""
    try { $mkStatus = (minikube status --format="{{.Host}}" 2>$null) } catch {}
    if ($mkStatus.Trim() -ne "Running") {
        Info "Starting Minikube (this may take a minute)..."
        minikube start --driver=docker --memory=4096 --cpus=2
        Ok "Minikube started."
    } else {
        Ok "Minikube already running."
    }

    # 3. Start minikube mount if not active
    #    Write a sentinel file on Windows, then verify it appears inside the VM.
    #    (ls /mnt/cs450 is unreliable — the dir exists even without an active mount)
    $sentinelPath = Join-Path $STORAGE_PATH ".mount_sentinel"
    Set-Content -Path $sentinelPath -Value "ok" -Encoding ascii

    $mountActive = $false
    try {
        $check = (minikube ssh "cat /mnt/cs450/.mount_sentinel 2>/dev/null") 2>$null
        $mountActive = ("$check".Trim() -eq "ok")
    } catch {
        $mountActive = $false
    }

    if (-not $mountActive) {
        Info "Starting minikube mount in a new window (keep that window open)..."
        Start-Process powershell -ArgumentList "-NoExit", "-Command", `
            "minikube mount '${STORAGE_PATH}:/mnt/cs450'"
        Info "Waiting 10 seconds for mount to establish..."
        Start-Sleep -Seconds 10

        # Verify sentinel is now visible from the VM
        try {
            $check = (minikube ssh "cat /mnt/cs450/.mount_sentinel 2>/dev/null") 2>$null
            if ("$check".Trim() -ne "ok") {
                Fail "Mount did not establish. Check the minikube mount window for errors."
            }
        } catch {
            Fail "Could not verify mount. Check the minikube mount window."
        }
        Ok "Mount active at /mnt/cs450."
    } else {
        Ok "Minikube mount already active."

    }

    # 4. Build worker Docker image inside Minikube if missing
    $imageCheck = ""
    try {
        $imageCheck = [string](minikube ssh "docker images video-analytics-worker --format '{{.Repository}}'" 2>$null)
    } catch {}

    if (-not $imageCheck -or (-not $imageCheck.Contains("video-analytics-worker"))) {
        Info "Building worker Docker image inside Minikube..."
        & minikube docker-env --shell powershell | Invoke-Expression
        docker build -t video-analytics-worker:latest -f container/Dockerfile .
        Ok "Worker image built."
    } else {
        Ok "Worker image already exists."
    }


    # 5. Create Gemini API key secret if missing
    $secretExists = $false
    try {
        kubectl get secret gemini-secret 2>&1 | Out-Null
        $secretExists = ($LASTEXITCODE -eq 0)
    } catch {
        $secretExists = $false
    }

    if (-not $secretExists) {
        $apiKey = if ($env:GEMINI_API_KEY) { $env:GEMINI_API_KEY } else { "" }
        if (-not $apiKey) {
            Warn "Creating empty Gemini secret. VLM will use fallback simulation."
        } else {
            Info "Creating Kubernetes Gemini API key secret..."
        }
        kubectl create secret generic gemini-secret --from-literal=api_key="$apiKey"
        Ok "Secret created."
    } else {
        Ok "Gemini secret already exists."
    }


    Info "Minikube setup complete."
}

# ===========================================================================
# Run the batch
# ===========================================================================
Info "Launching batch runner (workload: $workload, UseK8s: $UseK8s)..."

if ($UseK8s) {
    python run_batch.py --workload $workload --model $model --k8s --video-dir input_videos
} else {
    python run_batch.py --workload $workload --model $model --video-dir input_videos
}


if ($LASTEXITCODE -eq 0) {
    Ok "Batch complete. Results in serverless_results.csv, container_results.csv, and storage/outputs/."
} else {
    Fail "Batch exited with code $LASTEXITCODE."
}

