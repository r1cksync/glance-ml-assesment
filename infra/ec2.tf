# The single t3.small API host in the default VPC, fronted by an EIP.
#
# DEPENDENCY-CYCLE NOTE (CloudFront <-> EC2):
# CloudFront needs an origin DNS name; the instance's user_data must NOT need
# the CloudFront domain. We therefore:
#   1. allocate the EIP standalone (aws_eip.api — exists before everything),
#   2. point CloudFront at aws_eip.api.public_dns
#      (ec2-x-x-x-x.compute-1.amazonaws.com resolves publicly to the EIP),
#   3. pass IMAGE_BASE_URL="" to the API so it emits *relative* /images/<id>
#      URLs — the SAME CloudFront distribution serves /images/* from S3, so
#      relative URLs resolve correctly in the browser. No cycle anywhere.

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

# Latest Amazon Linux 2023 x86_64 (standard, not -minimal — the 2023* pattern
# excludes al2023-ami-minimal-*).
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023*-x86_64"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_security_group" "api" {
  name        = "${var.project}-api-sg"
  description = "fashion-retrieval API host"
  vpc_id      = data.aws_vpc.default.id

  # tcp/80 open to the world: CloudFront origin ranges cannot be IP-scoped
  # cheaply (no managed prefix-list free path worth the complexity here), and
  # the app itself requires a JWT for every non-public route.
  ingress {
    description = "HTTP from anywhere (CloudFront in front; app enforces JWT)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "all egress"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project}-api-sg"
  }
}

# Standalone EIP: created before the instance AND before CloudFront, breaking
# the CloudFront<->EC2 cycle. NOTE: public IPv4 costs ~$3.65/mo — unavoidable
# for any public endpoint.
resource "aws_eip" "api" {
  domain = "vpc"

  tags = {
    Name = "${var.project}-api"
  }
}

resource "aws_instance" "api" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  subnet_id              = sort(data.aws_subnets.default.ids)[0]
  vpc_security_group_ids = [aws_security_group.api.id]
  iam_instance_profile   = aws_iam_instance_profile.api.name

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }

  # "standard" credits: a runaway burst can never turn into surprise charges.
  credit_specification {
    cpu_credits = "standard"
  }

  metadata_options {
    http_tokens = "required" # IMDSv2 only
    # hop limit 2 so the API *container* (one extra network hop through the
    # docker bridge) can still reach instance-role credentials.
    http_put_response_hop_limit = 2
  }

  user_data = templatefile("${path.module}/ec2-user-data.sh.tpl", {
    aws_region       = var.aws_region
    ecr_registry     = local.ecr_registry
    ecr_image        = local.ecr_image
    log_group        = local.log_group
    s3_bucket        = aws_s3_bucket.data.bucket
    table_users      = aws_dynamodb_table.users.name
    table_tokens     = aws_dynamodb_table.tokens.name
    table_cache      = aws_dynamodb_table.parse_cache.name
    jwt_secret_param = aws_ssm_parameter.jwt_secret.name
    cors_origins     = local.cors_origins
    demo_password    = var.demo_password
    admin_email      = var.admin_email
    admin_password   = var.admin_password
  })
  user_data_replace_on_change = true

  # The awslogs docker driver needs the log group to exist
  # (awslogs-create-group=false), so order it explicitly.
  depends_on = [aws_cloudwatch_log_group.api]

  tags = {
    # CI deploys target this tag: aws ssm send-command
    #   --targets Key=tag:Name,Values=fashion-retrieval-api
    Name = "${var.project}-api"
  }
}

resource "aws_eip_association" "api" {
  instance_id   = aws_instance.api.id
  allocation_id = aws_eip.api.id
}
