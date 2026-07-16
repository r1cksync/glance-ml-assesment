# Infra — fashion-retrieval (us-east-1, AWS profile `glance`)

**Apply order** (single stack; Terraform sequences the internals — EIP before CloudFront before nothing, no cycles):

1. `terraform -chdir=infra init && terraform -chdir=infra apply` (or `make infra-up`)
2. `python scripts/seed_data.py --upload` — images to S3
3. Push the API image (CI on `main`, or manual `docker build -f api/Dockerfile` + push to `ecr_repo_url`), then `aws ssm send-command --targets Key=tag:Name,Values=fashion-retrieval-api --document-name AWS-RunShellScript --parameters 'commands=["systemctl restart fashion-api"]' --profile glance`
4. `bash scripts/deploy_frontend.sh` (Windows: `scripts/deploy_frontend.ps1`) — Amplify manual deploy

**Required vars:** none — everything is defaulted. Optional: `-var admin_email=... -var admin_password=...` (admin account created only when both set), `-var demo_password=...`, `-var aws_profile=""` for env-var credentials.

**Monthly cost:**

| Component | $/mo |
|---|---|
| EC2 t3.small (on-demand, standard credits) | 15.18 |
| Elastic IP (public IPv4) | 3.65 |
| S3 + DynamoDB on-demand + ECR (<1GB) + CloudFront (free tier) + Amplify + CloudWatch | ~1 combined at this scale |

**Teardown:** `bash scripts/teardown.sh` (Windows: `scripts/teardown.ps1`) — empties S3, deletes ECR images, then `terraform -chdir=infra destroy -auto-approve`.
