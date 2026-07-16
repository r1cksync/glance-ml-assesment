# Provider + shared locals. The stack is split into one .tf file per concern:
#   s3.tf dynamodb.tf ssm.tf ecr.tf iam.tf ec2.tf cloudfront.tf amplify.tf
#   cloudwatch.tf outputs.tf
#
# COST BUDGET (sacred): t3.small + EIP + CloudFront free tier + S3 pennies +
# DynamoDB on-demand + ECR <1GB + Amplify manual hosting + CloudWatch EMF.
# Nothing else. Everything here `terraform destroy`s cleanly (force_destroy /
# force_delete set where AWS would otherwise refuse to delete non-empty things).

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile != "" ? var.aws_profile : null

  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id

  bucket_name  = "${var.project}-${local.account_id}"
  table_users  = "${var.project}-users"
  table_tokens = "${var.project}-tokens"
  table_cache  = "${var.project}-parse-cache"
  log_group    = "/${var.project}/api"

  ecr_registry = "${local.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"
  ecr_image    = "${aws_ecr_repository.api.repository_url}:${var.api_image_tag}"

  # Amplify default domain for the manually deployed "main" branch.
  frontend_origin = "https://main.${aws_amplify_app.web.default_domain}"

  # The browser calls the API from the Amplify origin (CloudFront-served API is
  # same-origin for its own pages). localhost kept for local frontend dev.
  cors_origins = join(",", [local.frontend_origin, "http://localhost:3000"])
}
