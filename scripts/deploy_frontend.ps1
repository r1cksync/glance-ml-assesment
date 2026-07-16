# Build the static frontend against the deployed API URL and push it to
# Amplify Hosting via a manual deployment (PowerShell twin of deploy_frontend.sh).
#Requires -Version 7
param(
    [string]$AwsProfile = "glance",
    [string]$Region = "us-east-1"
)
$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$apiUrl = terraform -chdir=infra output -raw api_url
if ($LASTEXITCODE -ne 0) { throw "terraform output api_url failed - is the stack applied?" }
$appId = terraform -chdir=infra output -raw amplify_app_id
$frontendUrl = terraform -chdir=infra output -raw frontend_url

Write-Host ">>> building frontend against $apiUrl"
Push-Location frontend
try {
    npm ci
    if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
    $env:NEXT_PUBLIC_API_URL = $apiUrl
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "npm run build failed" }
} finally {
    Pop-Location
}

$zip = Join-Path $repoRoot "frontend/deploy.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Write-Host ">>> zipping frontend/out -> $zip"
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    (Join-Path $repoRoot "frontend/out"), $zip)

Write-Host ">>> creating amplify deployment (app: $appId, branch: main)"
$deploy = aws amplify create-deployment --app-id $appId --branch-name main `
    --profile $AwsProfile --region $Region | ConvertFrom-Json
if (-not $deploy.jobId) { throw "create-deployment returned no jobId" }

Write-Host ">>> uploading bundle"
Invoke-WebRequest -Method Put -Uri $deploy.zipUploadUrl -InFile $zip `
    -ContentType "application/zip" | Out-Null

Write-Host ">>> starting deployment job $($deploy.jobId)"
aws amplify start-deployment --app-id $appId --branch-name main `
    --job-id $deploy.jobId --profile $AwsProfile --region $Region | Out-Null

Write-Host ">>> waiting for job $($deploy.jobId)"
for ($i = 0; $i -lt 60; $i++) {
    $status = aws amplify get-job --app-id $appId --branch-name main `
        --job-id $deploy.jobId --profile $AwsProfile --region $Region `
        --query "job.summary.status" --output text
    Write-Host "    status: $status"
    if ($status -eq "SUCCEED") {
        Write-Host ">>> deployed: $frontendUrl"
        exit 0
    }
    if ($status -in @("FAILED", "CANCELLED")) { throw "deployment ended with status $status" }
    Start-Sleep -Seconds 5
}
throw "timed out waiting for deployment"
