# run_all.ps1

# 1. SETTINGS
$Workers = 10
$Limit = 100
$InputFolder = ".\datasets"
$OutputFolder = ".\results"
$PythonPath = ".\venv\Scripts\python.exe"

if (-not (Test-Path -Path $PythonPath)) {
    Write-Host "CRITICAL ERROR: Could not find Python in the venv at: $PythonPath" -ForegroundColor Red
    Write-Host "Please check the path. Are you in the project root?"
    exit
}
# Create output folder if it doesn't exist
if (-not (Test-Path -Path $OutputFolder)) {
    New-Item -ItemType Directory -Path $OutputFolder | Out-Null
}

# 2. FIND ALL CSV FILES
# We specifically ignore files that end in "_enriched.csv" to prevent double-processing
$files = Get-ChildItem -Path $InputFolder -Filter "*.csv" | Where-Object { $_.Name -notlike "*_enriched.csv" }

if ($files.Count -eq 0) {
    Write-Host "No valid CSV files found in $InputFolder (ignoring *_enriched.csv)." -ForegroundColor Red
    exit
}

Write-Host "Found $($files.Count) valid files to process." -ForegroundColor Cyan

# 3. LOOP THROUGH FILES
foreach ($file in $files) {
    $inputPath = $file.FullName
    $baseName = $file.BaseName
    $outputPath = Join-Path -Path $OutputFolder -ChildPath "${baseName}_results_${Limit}.csv"

    Write-Host "----------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host "STARTING: $baseName" -ForegroundColor Yellow
    Write-Host "Input:  $inputPath"
    Write-Host "Output: $outputPath"
    Write-Host "----------------------------------------------------------------"

    # 4. RUN THE PYTHON SCRIPT
    # We use Start-Process -Wait to ensure one finishes before the next starts
    # -NoNewWindow keeps it in the same terminal so you can see the progress bar
    & $PythonPath scripts/run_prompt_classifiers.py --input "$inputPath" --output "$outputPath" --limit $Limit --workers $Workers

    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: $baseName finished." -ForegroundColor Green
    } else {
        Write-Host "ERROR: $baseName failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }

    # Optional: Short cooldown between files to let the server breathe
    Start-Sleep -Seconds 5
}

Write-Host "----------------------------------------------------------------"
Write-Host "ALL JOBS COMPLETE. Time to wake up!" -ForegroundColor Cyan