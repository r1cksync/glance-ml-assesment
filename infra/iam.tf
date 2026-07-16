# Instance role for the API host. Deploys go through SSM Session Manager /
# RunCommand (AmazonSSMManagedInstanceCore) — no SSH keypair anywhere.

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "api" {
  name               = "${var.project}-api-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

data "aws_iam_policy_document" "api" {
  # Read-only S3 access to the data bucket (images + vector artifacts).
  statement {
    sid       = "S3Read"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/*"]
  }

  # CRUD on the three application tables.
  statement {
    sid = "DynamoCrud"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:UpdateItem",
      "dynamodb:DeleteItem",
      "dynamodb:Query",
      "dynamodb:Scan",
      "dynamodb:BatchGetItem",
      "dynamodb:BatchWriteItem",
      "dynamodb:ConditionCheckItem",
      "dynamodb:DescribeTable",
    ]
    resources = [
      aws_dynamodb_table.users.arn,
      aws_dynamodb_table.tokens.arn,
      aws_dynamodb_table.parse_cache.arn,
    ]
  }

  # JWT secret (and any future params under the project prefix).
  statement {
    sid     = "SsmParams"
    actions = ["ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath"]
    resources = [
      "arn:aws:ssm:${var.aws_region}:${local.account_id}:parameter/${var.project}/*",
    ]
  }

  # ECR pull: GetAuthorizationToken is account-wide by design.
  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "EcrPull"
    actions = [
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchCheckLayerAvailability",
    ]
    resources = [aws_ecr_repository.api.arn]
  }

  # awslogs docker log driver + EMF metrics-over-logs into the API log group.
  statement {
    sid = "Logs"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = [
      aws_cloudwatch_log_group.api.arn,
      "${aws_cloudwatch_log_group.api.arn}:*",
    ]
  }

  # Belt-and-braces: allow direct PutMetricData for the app namespace only
  # (EMF via the log group is the primary path and needs no extra permission).
  statement {
    sid       = "Metrics"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]

    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = ["FashionRetrieval"]
    }
  }
}

resource "aws_iam_role_policy" "api" {
  name   = "${var.project}-api-policy"
  role   = aws_iam_role.api.id
  policy = data.aws_iam_policy_document.api.json
}

# SSM agent permissions — enables Session Manager shells and RunCommand
# deploys (`aws ssm send-command ... systemctl restart fashion-api`).
resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.api.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "api" {
  name = "${var.project}-api-profile"
  role = aws_iam_role.api.name
}
