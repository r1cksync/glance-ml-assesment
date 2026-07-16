#!/usr/bin/env bash
# Build the static frontend against the deployed API URL and push it to
# Amplify Hosting via a manual deployment. Windows: scripts/deploy_frontend.ps1.
set -euo pipefail

PROFILE="${AWS_PROFILE_NAME:-glance}"
REGION="us-east-1"
PY="$(command -v python3 || command -v python)"

cd "$(dirname "$0")/.."

API_URL=$(terraform -chdir=infra output -raw api_url)
APP_ID=$(terraform -chdir=infra output -raw amplify_app_id)
FRONTEND_URL=$(terraform -chdir=infra output -raw frontend_url)

echo ">>> building frontend against $API_URL"
pushd frontend >/dev/null
npm ci
NEXT_PUBLIC_API_URL="$API_URL" npm run build
popd >/dev/null

ZIP="frontend/deploy.zip"
rm -f "$ZIP"
echo ">>> zipping frontend/out -> $ZIP"
(cd frontend/out && zip -qr ../deploy.zip .)

echo ">>> creating amplify deployment (app: $APP_ID, branch: main)"
DEPLOY_JSON=$(aws amplify create-deployment --app-id "$APP_ID" --branch-name main \
  --profile "$PROFILE" --region "$REGION")
JOB_ID=$(printf '%s' "$DEPLOY_JSON" | "$PY" -c "import json,sys; print(json.load(sys.stdin)['jobId'])")
UPLOAD_URL=$(printf '%s' "$DEPLOY_JSON" | "$PY" -c "import json,sys; print(json.load(sys.stdin)['zipUploadUrl'])")

echo ">>> uploading bundle"
curl -sf -H "Content-Type: application/zip" -T "$ZIP" "$UPLOAD_URL"

echo ">>> starting deployment job $JOB_ID"
aws amplify start-deployment --app-id "$APP_ID" --branch-name main --job-id "$JOB_ID" \
  --profile "$PROFILE" --region "$REGION" >/dev/null

echo ">>> waiting for job $JOB_ID"
for _ in $(seq 1 60); do
  STATUS=$(aws amplify get-job --app-id "$APP_ID" --branch-name main --job-id "$JOB_ID" \
    --profile "$PROFILE" --region "$REGION" --query 'job.summary.status' --output text)
  echo "    status: $STATUS"
  case "$STATUS" in
    SUCCEED)
      echo ">>> deployed: $FRONTEND_URL"
      exit 0
      ;;
    FAILED|CANCELLED)
      echo ">>> deployment ended with status $STATUS" >&2
      exit 1
      ;;
  esac
  sleep 5
done

echo ">>> timed out waiting for deployment" >&2
exit 1
