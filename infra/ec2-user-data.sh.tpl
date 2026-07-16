#!/usr/bin/env bash
# Rendered by Terraform templatefile(). Dollar-brace sequences are Terraform
# interpolations; runtime shell variables are deliberately unbraced ($VAR)
# because templatefile leaves a bare dollar sign alone.
set -euxo pipefail

MARKER=/var/lib/fashion-api/.bootstrapped
mkdir -p /var/lib/fashion-api
if [ -f "$MARKER" ]; then
  echo "fashion-api already bootstrapped - nothing to do"
  exit 0
fi

# ── docker ───────────────────────────────────────────────────────────────────
dnf install -y docker
systemctl enable --now docker

# ── 3GB swap: t3.small has 2GB RAM; swap absorbs model-load spikes ───────────
if ! grep -q '^/swapfile' /etc/fstab; then
  dd if=/dev/zero of=/swapfile bs=1M count=3072
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# ── container environment ────────────────────────────────────────────────────
cat > /etc/fashion-api.env <<'ENVEOF'
S3_BUCKET=${s3_bucket}
TABLE_USERS=${table_users}
TABLE_TOKENS=${table_tokens}
TABLE_CACHE=${table_cache}
JWT_SECRET_SSM_PARAM=${jwt_secret_param}
# IMAGE_BASE_URL is INTENTIONALLY empty: the API emits relative "/images/<id>"
# URLs and the SAME CloudFront distribution that fronts the API serves
# /images/* from S3 — so relative URLs resolve perfectly in the browser.
# This breaks the CloudFront <-> EC2 circular dependency (the CloudFront
# domain does not exist yet when this instance's user_data is rendered).
IMAGE_BASE_URL=
CORS_ORIGINS=${cors_origins}
AWS_REGION=${aws_region}
AWS_DEFAULT_REGION=${aws_region}
# Rerank is a GPU-tier quality stage: BLIP-ITM over 50 candidates costs
# 15-40s/query on a t3.small CPU (and needs local image files). It stays in
# the offline eval/ablation story; API exposes use_rerank per-request for
# environments that can afford it.
USE_RERANK=false
DEMO_USER_EMAIL=demo@fashionretrieval.dev
DEMO_USER_PASSWORD=${demo_password}
%{ if admin_email != "" }
ADMIN_EMAIL=${admin_email}
%{ endif }
%{ if admin_password != "" }
ADMIN_PASSWORD=${admin_password}
%{ endif }
ENVEOF
chmod 600 /etc/fashion-api.env

# ── ECR login + pull helper ──────────────────────────────────────────────────
# Retries up to 10x30s: on the very first boot the image may not have been
# pushed to ECR yet (CI builds it after terraform apply).
cat > /usr/local/bin/fashion-api-pull.sh <<'PULLEOF'
#!/usr/bin/env bash
set -uo pipefail
REGISTRY="${ecr_registry}"
IMAGE="${ecr_image}"
REGION="${aws_region}"

for i in $(seq 1 10); do
  aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$REGISTRY" || true
  if docker pull "$IMAGE"; then
    exit 0
  fi
  echo "pull attempt $i/10 failed; retrying in 30s"
  sleep 30
done

# Fall back to a previously pulled image if one is cached locally.
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "pull failed but a cached image exists - starting with it"
  exit 0
fi
echo "no image available in ECR or local cache"
exit 1
PULLEOF
chmod +x /usr/local/bin/fashion-api-pull.sh

# ── systemd unit ─────────────────────────────────────────────────────────────
# Deploys are just: docker image updated in ECR + `systemctl restart
# fashion-api` (via SSM RunCommand) — ExecStartPre re-pulls :latest.
cat > /etc/systemd/system/fashion-api.service <<'UNITEOF'
[Unit]
Description=Fashion Retrieval API (docker)
After=docker.service network-online.target
Requires=docker.service
Wants=network-online.target

[Service]
Type=simple
Restart=always
RestartSec=15
TimeoutStartSec=900
ExecStartPre=/usr/local/bin/fashion-api-pull.sh
ExecStartPre=-/usr/bin/docker rm -f api
ExecStart=/usr/bin/docker run --rm --name api -p 80:8000 \
  --env-file /etc/fashion-api.env \
  -v /var/lib/fashion-api/models:/models \
  -v /var/lib/fashion-api/artifacts:/app/data/artifacts \
  --log-driver=awslogs \
  --log-opt awslogs-group=${log_group} \
  --log-opt awslogs-region=${aws_region} \
  --log-opt awslogs-create-group=false \
  --log-opt awslogs-stream=api \
  ${ecr_image}
ExecStop=/usr/bin/docker stop -t 30 api

[Install]
WantedBy=multi-user.target
UNITEOF

systemctl daemon-reload
systemctl enable fashion-api.service
# --no-block: do not fail user_data if the image is not in ECR yet; the unit
# keeps retrying (Restart=always) until CI pushes the first image.
systemctl start --no-block fashion-api.service

touch "$MARKER"
echo "fashion-api bootstrap complete"
