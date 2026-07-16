#!/usr/bin/env bash
# Build + push the API image via AWS CodeBuild (no local Docker needed).
# Usage: bash scripts/build_image.sh [profile] [region]
set -euo pipefail

PROFILE="${1:-glance}"
REGION="${2:-us-east-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUCKET=$(terraform -chdir="$ROOT/infra" output -raw bucket_name)
PROJECT_NAME="fashion-retrieval-api-build"

echo ">> zipping source (tracked files only)"
cd "$ROOT"
git archive --format=zip -o /tmp/fashion-source.zip HEAD

echo ">> uploading to s3://$BUCKET/build/source.zip"
aws s3 cp /tmp/fashion-source.zip "s3://$BUCKET/build/source.zip" \
  --profile "$PROFILE" --region "$REGION" --only-show-errors

echo ">> starting CodeBuild"
BUILD_ID=$(aws codebuild start-build --project-name "$PROJECT_NAME" \
  --profile "$PROFILE" --region "$REGION" \
  --query 'build.id' --output text)
echo "   build: $BUILD_ID"

while true; do
  STATUS=$(aws codebuild batch-get-builds --ids "$BUILD_ID" \
    --profile "$PROFILE" --region "$REGION" \
    --query 'builds[0].buildStatus' --output text)
  echo "   status: $STATUS"
  case "$STATUS" in
    SUCCEEDED) echo ">> image pushed"; exit 0 ;;
    FAILED|FAULT|STOPPED|TIMED_OUT) echo ">> build failed: $STATUS"; exit 1 ;;
  esac
  sleep 20
done
