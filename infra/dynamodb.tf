# Three on-demand (PAY_PER_REQUEST) tables — pennies at this scale.

resource "aws_dynamodb_table" "users" {
  name         = local.table_users
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "email"

  attribute {
    name = "email"
    type = "S"
  }
}

resource "aws_dynamodb_table" "tokens" {
  name         = local.table_tokens
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "jti_hash"

  attribute {
    name = "jti_hash"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}

resource "aws_dynamodb_table" "parse_cache" {
  name         = local.table_cache
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "cache_key"

  attribute {
    name = "cache_key"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}
