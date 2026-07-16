# Amplify Hosting in MANUAL deployment mode: no repository connected; the
# deploy scripts (scripts/deploy_frontend.*) zip the static Next.js export and
# push it with `aws amplify create-deployment` / `start-deployment`.
#
# NOTE: NEXT_PUBLIC_API_URL is baked into the static bundle at BUILD time by
# the deploy script (from `terraform output api_url`) — Amplify env vars play
# no role for manual static deployments.

resource "aws_amplify_app" "web" {
  name     = "${var.project}-web"
  platform = "WEB"

  # SPA fallback: deep links into the static export rewrite to index.html.
  custom_rule {
    source = "/<*>"
    status = "404-200"
    target = "/index.html"
  }
}

resource "aws_amplify_branch" "main" {
  app_id      = aws_amplify_app.web.id
  branch_name = "main"
  stage       = "PRODUCTION"
}
