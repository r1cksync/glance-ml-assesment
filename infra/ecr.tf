# Container registry for the API image. Lifecycle policy keeps the repo
# comfortably under the 1GB budget line (last 5 images only).

resource "aws_ecr_repository" "api" {
  name = "${var.project}-api"

  # Big-red-button friendly: destroy deletes images too.
  force_delete = true

  image_scanning_configuration {
    scan_on_push = false
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "keep only the last 5 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 5
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}
