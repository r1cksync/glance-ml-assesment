# CodeBuild project that builds the API Docker image INSIDE AWS and pushes it
# to ECR. Source = a zip of the repo uploaded to s3://<bucket>/build/source.zip
# by scripts/build_image.sh. This keeps multi-GB docker builds off the dev
# machine and gives CI-independent deploys (GitHub Actions uses the same ECR).
# Cost: general1.small @ $0.005/min — a ~10 min build is ~5 cents (first
# 100 min/mo free on new accounts).

resource "aws_codebuild_project" "api_image" {
  name          = "${var.project}-api-build"
  description   = "docker build api/Dockerfile -> ECR :latest"
  service_role  = aws_iam_role.codebuild.arn
  build_timeout = 30

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type                = "BUILD_GENERAL1_SMALL"
    image                       = "aws/codebuild/standard:7.0"
    type                        = "LINUX_CONTAINER"
    privileged_mode             = true # docker-in-docker

    environment_variable {
      name  = "ECR_REPO"
      value = aws_ecr_repository.api.repository_url
    }
    environment_variable {
      name  = "AWS_ACCOUNT_ID"
      value = data.aws_caller_identity.current.account_id
    }
  }

  source {
    type     = "S3"
    location = "${aws_s3_bucket.data.bucket}/build/source.zip"
    buildspec = <<-SPEC
      version: 0.2
      phases:
        pre_build:
          commands:
            - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com
        build:
          commands:
            - docker build -f api/Dockerfile -t $ECR_REPO:latest .
        post_build:
          commands:
            - docker push $ECR_REPO:latest
    SPEC
  }

  logs_config {
    cloudwatch_logs {
      group_name  = aws_cloudwatch_log_group.api.name
      stream_name = "codebuild"
    }
  }

  tags = { Project = var.project }
}

data "aws_iam_policy_document" "codebuild_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["codebuild.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "codebuild" {
  name               = "${var.project}-codebuild"
  assume_role_policy = data.aws_iam_policy_document.codebuild_assume.json
}

data "aws_iam_policy_document" "codebuild" {
  statement {
    sid = "Logs"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["${aws_cloudwatch_log_group.api.arn}:*", aws_cloudwatch_log_group.api.arn]
  }

  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "EcrPush"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:CompleteLayerUpload",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [aws_ecr_repository.api.arn]
  }

  statement {
    sid       = "SourceZip"
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${aws_s3_bucket.data.arn}/build/*"]
  }
}

resource "aws_iam_role_policy" "codebuild" {
  name   = "${var.project}-codebuild"
  role   = aws_iam_role.codebuild.id
  policy = data.aws_iam_policy_document.codebuild.json
}
