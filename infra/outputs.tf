# Stack outputs consumed by the deploy scripts, CI, and humans.

output "api_url" {
  description = "Public HTTPS API endpoint (CloudFront in front of the EC2 API; also serves /images/* from S3)."
  value       = "https://${aws_cloudfront_distribution.main.domain_name}"
}

output "frontend_url" {
  description = "Amplify-hosted frontend URL (main branch, manual deployments)."
  value       = "https://main.${aws_amplify_app.web.default_domain}"
}

output "amplify_app_id" {
  description = "Amplify app id for scripts/deploy_frontend.*."
  value       = aws_amplify_app.web.id
}

output "ecr_repo_url" {
  description = "ECR repository URL for the API image."
  value       = aws_ecr_repository.api.repository_url
}

output "bucket_name" {
  description = "S3 data bucket (images/ + vectors/)."
  value       = aws_s3_bucket.data.bucket
}

output "instance_id" {
  description = "API EC2 instance id (SSM Session Manager target)."
  value       = aws_instance.api.id
}

output "dashboard_url" {
  description = "CloudWatch dashboard deep link."
  value       = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.main.dashboard_name}"
}

output "eip" {
  description = "Elastic IP of the API host (CloudFront origin resolves to this)."
  value       = aws_eip.api.public_ip
}
