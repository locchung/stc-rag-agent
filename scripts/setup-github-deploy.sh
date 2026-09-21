#!/usr/bin/env bash
# Thiết lập MỘT LẦN để GitHub Actions deploy được lên Cloud Run, không cần khoá JSON.
#
# Tạo một service account riêng cho GitHub với quyền tối thiểu, và một Workload
# Identity Federation chỉ tin đúng repo này, đúng nhánh master. Cuối script in ra hai
# giá trị cần đặt làm Variables trong GitHub (Settings -> Secrets and variables).
#
#   bash scripts/setup-github-deploy.sh
set -euo pipefail

PROJECT_ID=seatecco-rag
REGION=asia-southeast1
AR_REPO=seatecco
GITHUB_REPO=locchung/stc-rag-agent
SA_NAME=github-deployer
POOL=github
PROVIDER=github-actions

PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
SA="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
RUNTIME_SA="$PROJECT_NUMBER-compute@developer.gserviceaccount.com"   # SA mà service chạy dưới danh nghĩa

gcloud services enable iamcredentials.googleapis.com sts.googleapis.com --project "$PROJECT_ID"

# 1. Service account riêng cho GitHub
gcloud iam service-accounts create "$SA_NAME" --project "$PROJECT_ID" \
  --display-name "GitHub Actions deploy seatecco-rag"

# 2. Quyền tối thiểu.
#    run.developer: deploy và chuyển traffic, nhưng KHÔNG sửa được IAM của service -
#    tức là không tự mở service ra public được.
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:$SA" --role roles/run.developer --condition=None
#    Chỉ push được vào ĐÚNG repository image này, không phải mọi repository trong project.
gcloud artifacts repositories add-iam-policy-binding "$AR_REPO" \
  --project "$PROJECT_ID" --location "$REGION" \
  --member "serviceAccount:$SA" --role roles/artifactregistry.writer
#    Deploy một service chạy dưới danh nghĩa RUNTIME_SA thì phải được "act as" nó.
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" --project "$PROJECT_ID" \
  --member "serviceAccount:$SA" --role roles/iam.serviceAccountUser

# 3. Pool + provider OIDC. attribute-condition là chỗ QUAN TRỌNG NHẤT: thiếu nó thì
#    workflow của BẤT KỲ repo GitHub nào cũng đổi token lấy quyền deploy được.
gcloud iam workload-identity-pools create "$POOL" --project "$PROJECT_ID" \
  --location global --display-name "GitHub Actions"
gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --project "$PROJECT_ID" \
  --location global --workload-identity-pool "$POOL" \
  --issuer-uri "https://token.actions.githubusercontent.com" \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition "assertion.repository == '$GITHUB_REPO' && assertion.ref == 'refs/heads/master'"

# 4. Cho workflow của repo đó mượn service account
gcloud iam service-accounts add-iam-policy-binding "$SA" --project "$PROJECT_ID" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/attribute.repository/$GITHUB_REPO"

echo
echo "Đặt hai Variables này trong GitHub (Settings -> Secrets and variables -> Actions -> Variables):"
echo "  GCP_WIF_PROVIDER = projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/providers/$PROVIDER"
echo "  GCP_DEPLOY_SA    = $SA"
echo
echo "Và một Secret:"
echo "  GOOGLE_API_KEY   = key Gemini (dùng để dựng index trong CI)"
