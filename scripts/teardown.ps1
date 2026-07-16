# Big red button (PowerShell twin of teardown.sh): empty the S3 bucket,
# delete ECR images, then destroy the whole terraform stack. Safe to re-run.
#Requires -Version 7
param(
    [string]$AwsProfile = "glance",
    [string]$Region = "us-east-1",
    [string]$Project = "fashion-retrieval"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host ">>> resolving AWS account (profile: $AwsProfile)"
$account = aws sts get-caller-identity --profile $AwsProfile --query Account --output text
if ($LASTEXITCODE -ne 0) { throw "cannot resolve AWS account for profile $AwsProfile" }
$bucket = "$Project-$account"

Write-Host ">>> emptying s3://$bucket"
aws s3 rm "s3://$bucket" --recursive --profile $AwsProfile --region $Region 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "    (bucket missing or already empty - continuing)" }

Write-Host ">>> deleting all images in ECR repo $Project-api"
$digests = aws ecr list-images --repository-name "$Project-api" `
    --profile $AwsProfile --region $Region `
    --query "imageIds[*].imageDigest" --output text 2>$null
if ($LASTEXITCODE -eq 0 -and $digests -and $digests -ne "None") {
    foreach ($digest in ($digests -split "\s+" | Where-Object { $_ })) {
        aws ecr batch-delete-image --repository-name "$Project-api" `
            --image-ids "imageDigest=$digest" `
            --profile $AwsProfile --region $Region | Out-Null
    }
} else {
    Write-Host "    (no images found)"
}

Write-Host ">>> terraform destroy (infra/)"
terraform -chdir=infra destroy -auto-approve -var "aws_profile=$AwsProfile"
if ($LASTEXITCODE -ne 0) { throw "terraform destroy failed" }

Write-Host ">>> done - all $Project infrastructure destroyed"
