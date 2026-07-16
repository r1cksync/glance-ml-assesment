# Input variables. Everything is defaulted so `terraform apply` works out of
# the box with the human-operator AWS profile "glance" in us-east-1.

variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Named AWS CLI profile used by human operators. Set to \"\" to fall back to standard env-var credential resolution (e.g. in CI)."
  type        = string
  default     = "glance"
}

variable "project" {
  description = "Project prefix applied to every resource name."
  type        = string
  default     = "fashion-retrieval"
}

variable "instance_type" {
  description = "EC2 instance type for the API host. Budget: t3.small (~$15.18/mo on-demand)."
  type        = string
  default     = "t3.small"
}

variable "api_image_tag" {
  description = "ECR image tag the API host runs."
  type        = string
  default     = "latest"
}

variable "admin_email" {
  description = "Admin user email. The API creates the admin account only when both admin_email and admin_password are non-empty."
  type        = string
  default     = ""
}

variable "admin_password" {
  description = "Admin user password. Leave empty to skip admin account creation."
  type        = string
  default     = ""
  sensitive   = true
}

variable "demo_password" {
  description = "Password for the built-in demo user (demo@fashionretrieval.dev)."
  type        = string
  default     = "DemoPass123!"
}
