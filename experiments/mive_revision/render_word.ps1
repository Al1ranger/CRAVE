param(
    [string]$Manuscript = 'D:\Study\Papers\p6\MIVE_Verified_Manuscript.docx',
    [string]$OutputDirectory = 'D:\Study\Papers\p6\experiments\mive_revision\rendered'
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$taskPdf = Join-Path $OutputDirectory 'MIVE_Verified_Manuscript.pdf'
$taskWord = $null
$taskDocument = $null
$taskOwnsWord = $false
try {
    $taskWord = New-Object -ComObject Word.Application
    if ($taskWord.Documents.Count -gt 0) { throw 'Refusing to alter a Word instance containing open user documents.' }
    $taskOwnsWord = $true
    $taskWord.Visible = $false
    $taskWord.DisplayAlerts = 0
    $taskDocument = $taskWord.Documents.Open($Manuscript, $false, $true)
    $taskDocument.Repaginate()
    $taskDocument.Fields.Update() | Out-Null
    $taskDocument.ExportAsFixedFormat($taskPdf, 17)
    @{pdf=$taskPdf;pages=$taskDocument.ComputeStatistics(2)} | ConvertTo-Json -Compress
} finally {
    if ($null -ne $taskDocument) {
        $taskDocument.Close(0)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($taskDocument)
    }
    if ($null -ne $taskWord) {
        if ($taskOwnsWord) { $taskWord.Quit() }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($taskWord)
    }
}
