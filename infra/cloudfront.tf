# One CloudFront distribution, two origins:
#   default behavior  -> EC2 API (custom origin over http on the EIP's public
#                        DNS name — the EIP exists before this distribution,
#                        which is what breaks the CloudFront<->EC2 cycle)
#   /images/*         -> S3 via Origin Access Control
#
# S3 keys already carry the images/ prefix (images/<image_id>), and the API
# links "/images/<image_id>" — so origin_path stays "" and the viewer path
# maps onto the S3 key 1:1. Free tier: 1TB egress + 10M requests/mo.

locals {
  # AWS managed cache/origin-request policies (fixed well-known IDs).
  cache_policy_caching_disabled       = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad" # Managed-CachingDisabled
  cache_policy_caching_optimized      = "658327ea-f89d-4fab-a63d-7e88639e58f6" # Managed-CachingOptimized
  origin_policy_all_viewer_except_host = "b689b0a8-53d0-40ab-baf2-68738e2966ac" # Managed-AllViewerExceptHostHeader

  api_origin_id    = "ec2-api"
  images_origin_id = "s3-images"
}

resource "aws_cloudfront_origin_access_control" "images" {
  name                              = "${var.project}-images-oac"
  description                       = "OAC for /images/* -> private S3 bucket"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_cloudfront_distribution" "main" {
  enabled         = true
  comment         = "${var.project}: API + images CDN"
  price_class     = "PriceClass_100"
  is_ipv6_enabled = true

  # ── API origin: the EIP's public DNS (resolves to the EIP from outside) ──
  origin {
    domain_name = aws_eip.api.public_dns
    origin_id   = local.api_origin_id

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "http-only" # TLS terminates at CloudFront
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  # ── images origin: private S3, reachable only through this distribution ──
  origin {
    domain_name              = aws_s3_bucket.data.bucket_regional_domain_name
    origin_id                = local.images_origin_id
    origin_access_control_id = aws_cloudfront_origin_access_control.images.id
    # origin_path deliberately empty: S3 keys are images/<id> and viewers
    # request /images/<id> — a 1:1 match.
    origin_path = ""
  }

  # ── default: the API (no caching, forward everything except Host) ────────
  default_cache_behavior {
    target_origin_id       = local.api_origin_id
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    compress               = true

    cache_policy_id          = local.cache_policy_caching_disabled
    origin_request_policy_id = local.origin_policy_all_viewer_except_host
  }

  # ── /images/*: immutable dataset images, cache hard ──────────────────────
  ordered_cache_behavior {
    path_pattern           = "/images/*"
    target_origin_id       = local.images_origin_id
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    compress               = true

    cache_policy_id = local.cache_policy_caching_optimized
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

# Bucket policy: only this distribution may read, and only under images/*
# (vectors/* stays fully private).
data "aws_iam_policy_document" "cdn_read" {
  statement {
    sid     = "AllowCloudFrontOACRead"
    actions = ["s3:GetObject"]

    principals {
      type        = "Service"
      identifiers = ["cloudfront.amazonaws.com"]
    }

    resources = ["${aws_s3_bucket.data.arn}/images/*"]

    condition {
      test     = "StringEquals"
      variable = "AWS:SourceArn"
      values   = [aws_cloudfront_distribution.main.arn]
    }
  }
}

resource "aws_s3_bucket_policy" "cdn_read" {
  bucket = aws_s3_bucket.data.id
  policy = data.aws_iam_policy_document.cdn_read.json

  depends_on = [aws_s3_bucket_public_access_block.data]
}
