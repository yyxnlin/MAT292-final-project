param (
    [string[]]$Steps = @("data_stats", "combine", "fhn", "filter", "filtered_fhn_stats", "balance", "model"),
    [string]$Categories = "binary",
    [string[]]$Features = @("pt_width", "qrs_width"),
    [float]$LossThreshold = 0.1,
    [float]$R2Threshold = 0.8,
    [string]$DataDir = "data",
    [string]$OutputDir = "output",
    [string]$PlotsDir = "config_3_width_only"
)

Set-Location (Split-Path $PSScriptRoot -Parent)

# --- Internal Configuration ---
$env:PYTHONWARNINGS = "ignore"
$RawDataPath = Join-Path $OutputDir "all_waves_raw.parquet"
$FhnDataPath = Join-Path $OutputDir "all_fhn_data_raw.parquet"

Write-Host "-------------------------------------------------------" -ForegroundColor Cyan
Write-Host "ECG Classification Pipeline Automator"
Write-Host "-------------------------------------------------------"

# 1. Dependency Check
if (Test-Path "requirements.txt") {
    Write-Host "[1/4] Checking dependencies..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt --quiet
}

# 2. Skipping steps if cached files already exist
if ($Steps.Count -eq 7) { 
    if (Test-Path $FhnDataPath) {
        Write-Host "[i] Detected existing FHN data. Starting from 'filter' to save time." -ForegroundColor Gray
        $Steps = @("data_stats", "filter", "balance", "filtered_fhn_stats", "model")
    } elseif (Test-Path $RawDataPath) {
        Write-Host "[i] Detected existing raw waves. Starting from 'fhn'." -ForegroundColor Gray
        $Steps = @("data_stats", "fhn", "filter", "balance", "filtered_fhn_stats", "model")
    }
}

# 3. Create Directories
if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }
if (-not (Test-Path $PlotsDir)) { New-Item -ItemType Directory -Path $PlotsDir | Out-Null }

# 4. Run the Pipeline
Write-Host "[2/4] Executing: $($Steps -join ', ')" -ForegroundColor Yellow
Write-Host "[3/4] Processing..." -ForegroundColor Yellow

python -W ignore pipeline.py --step $Steps `
    --data_folder $DataDir `
    --output_folder $OutputDir `
    --plots_folder $PlotsDir `
    --categories $Categories `
    --features $Features `
    --r2_threshold $R2Threshold `
    --loss_threshold $LossThreshold 


# 5. Final Check
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[4/4] Pipeline completed successfully!" -ForegroundColor Green
    Write-Host "Results are in: $OutputDir and $PlotsDir" -ForegroundColor Gray
} else {
    Write-Host "`n[!] Pipeline failed. Check the error log or verify your data folder." -ForegroundColor Red
}