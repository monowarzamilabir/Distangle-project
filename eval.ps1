# Finish the Qwen3-4B evaluation.  Run from D:\791 in PowerShell:
#     .\eval.ps1
#
# Resumes from completed item ids, so stopping it with Ctrl+C and starting
# it again never repeats work and never corrupts the predictions file.
#
# Uses the bengali-rq2 environment explicitly: the base anaconda python has
# matplotlib but no torch, so plain `python eval_math.py` will fail.

$py = "C:\anaconda\envs\bengali-rq2\python.exe"
$env:PYTHONIOENCODING = "utf-8"

foreach ($cond in @("before", "after")) {
    Write-Host ""
    Write-Host "=== qwen3-4b $cond ===" -ForegroundColor Cyan
    & $py -u eval_math.py --model qwen3-4b --condition $cond
    if ($LASTEXITCODE -ne 0) {
        Write-Host "qwen3-4b $cond exited $LASTEXITCODE, stopping." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host ""
Write-Host "All evaluation complete. Now run .\rebuild.ps1" -ForegroundColor Green
