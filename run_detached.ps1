# Launch the eval directly under python, no bash wrapper. The bash -lc form
# started a login shell that changed directory and never reached the script.
$py  = "C:\anaconda\envs\bengali-rq2\python.exe"
$log = "D:\791\logs\qwen4b_250.log"
$env:PYTHONIOENCODING = "utf-8"

foreach ($cond in @("before", "after")) {
    Add-Content $log "[$(Get-Date -Format HH:mm:ss)] === qwen3-4b $cond (detached) ==="
    $p = Start-Process -FilePath $py `
        -ArgumentList "-u", "eval_math.py", "--model", "qwen3-4b", "--condition", $cond `
        -WorkingDirectory "D:\791" -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "D:\791\logs\_q4b_$cond.out" `
        -RedirectStandardError  "D:\791\logs\_q4b_$cond.err"
    Get-Content "D:\791\logs\_q4b_$cond.out" -ErrorAction SilentlyContinue | Add-Content $log
    Get-Content "D:\791\logs\_q4b_$cond.err" -ErrorAction SilentlyContinue | Add-Content $log
    Add-Content $log "[$(Get-Date -Format HH:mm:ss)] qwen3-4b $cond exit $($p.ExitCode)"
}
Add-Content $log "[$(Get-Date -Format HH:mm:ss)] === ALL EVALUATION COMPLETE ==="
