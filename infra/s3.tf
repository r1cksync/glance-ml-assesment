# One private bucket for everything: images/<image_id> (served through
# CloudFront via OAC) and vectors/* (index + onnx artifacts, private).

resource "aws_s3_bucket" "data" {
  bucket = local.bucket_name

  # Big-red-button friendly: destroy deletes objects too.
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
