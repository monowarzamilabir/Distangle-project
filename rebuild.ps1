# Rebuild every paper number, figure and section from the stored model
# outputs.  Run from D:\791 in PowerShell once the evaluation is done:
#     .\rebuild.ps1
#
# These three use the base anaconda python; they only need matplotlib,
# numpy and the standard library, not torch.
#
# Safe to run any time, including mid-evaluation. It reads whatever
# predictions exist and reports each model on the items it has finished
# in both conditions.

foreach ($step in @(
    @("draft_results.py", "accuracy table, McNemar tests, tab_main.tex"),
    @("make_figs.py",     "fig_scale, fig_beforeafter, fig_difficulty"),
    @("merge_paper.py",   "paper/paper_body.tex"))) {

    Write-Host ""
    Write-Host ("=== " + $step[0] + "  ->  " + $step[1] + " ===") -ForegroundColor Cyan
    python $step[0]
    if ($LASTEXITCODE -ne 0) {
        Write-Host ($step[0] + " failed, stopping.") -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host ""
Write-Host "Done. Re-upload these to Overleaf:" -ForegroundColor Green
Write-Host "  paper\figs\fig_scale.png"
Write-Host "  paper\figs\fig_beforeafter.png"
Write-Host "  paper\figs\fig_difficulty.png"
Write-Host "  paper\paper_body.tex   (re-paste)"
Write-Host ""
Write-Host "Then check by hand the two places Qwen3-4B is quoted in prose:" -ForegroundColor Yellow
Write-Host "  paper\front_matter.tex   abstract, the 45.0% in the scale ladder"
Write-Host "  paper\paper_body.tex     Results, the 45.0 to 38.0 sentence"
