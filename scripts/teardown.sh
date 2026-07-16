#!/usr/bin/env bash
# Big red button: empty the S3 bucket, delete ECR images, then destroy the
# whole terraform stack. Safe to re-run. Windows users: scripts/teardown.ps1.
set -euo pipefail

PROFILE="${AWS_PROFILE_NAME:-glance}"
REGION="us-east-1"
PROJECT="fashion-retrieval"

cd "$(dirname "$0")/.."

echo ">>> resolving AWS account (profile: $PROFILE)"
ACCOUNT=$(aws sts get-caller-identity --profile "$PROFILE" --query Account --output text)
BUCKET="$PROJECT-$ACCOUNT"

echo ">>> emptying s3://$BUCKET"
aws s3 rm "s3://$BUCKET" --recursive --profile "$PROFILE" --region "$REGION" \
  || echo "    (bucket missing or already empty — continuing)"

echo ">>> deleting all images in ECR repo $PROJECT-api"
IMAGE_IDS=$(aws ecr list-images --repository-name "$PROJECT-api" \
  --profile "$PROFILE" --region "$REGION" \
  --query 'imageIds[*]' --output json 2>/dev/null || echo '[]')
if [ -n "$IMAGE_IDS" ] && [ "$IMAGE_IDS" != "[]" ]; then
  aws ecr batch-delete-image --repository-name "$PROJECT-api" \
    --image-ids "$IMAGE_IDS" --profile "$PROFILE" --region "$REGION" >/dev/null \
    || echo "    (image delete failed — terraform force_delete will handle it)"
else
  echo "    (no images found)"
fi

echo ">>> terraform destroy (infra/)"
terraform -chdir=infra destroy -auto-approve -var "aws_profile=$PROFILE"

echo ">>> done — all $PROJECT infrastructure destroyed"
