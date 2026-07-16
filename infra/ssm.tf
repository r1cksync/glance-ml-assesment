# JWT signing secret: generated once by Terraform, stored as a SecureString.
# The API reads it at boot via ssm:GetParameter (instance role).

resource "random_password" "jwt_secret" {
  length  = 64
  special = false # keeps the value shell/env-file safe
}

resource "aws_ssm_parameter" "jwt_secret" {
  name        = "/${var.project}/jwt-secret"
  description = "HMAC secret for API JWTs."
  type        = "SecureString"
  value       = random_password.jwt_secret.result
}
