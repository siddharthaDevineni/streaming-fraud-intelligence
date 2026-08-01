#!/bin/bash
# Deploys the full Streaming Fraud Intelligence stack to GKE.
# Usage: ./k8s/deploy.sh
#
# Idempotent - safe to re-run; kubectl apply only changes what differs.
# Also self-healing: creates the GKE cluster if it doesn't already exist.

set -e

# Always run from the project root, regardless of where this script is invoked from
cd "$(dirname "$0")/.."

PROJECT=streaming-fraud-intelligence
REGION=europe-west3
NAMESPACE=fraud-detection
CLUSTER=streaming-fraud-intel

echo "=================================================="
echo "  Deploying Streaming Fraud Intelligence to GKE"
echo "=================================================="

echo ""
echo "[0/10] Checking if cluster exists..."
if ! gcloud container clusters describe $CLUSTER \
    --location=$REGION --project=$PROJECT &> /dev/null; then
  echo "    Cluster '$CLUSTER' not found — creating (5-10 min)..."
  gcloud container clusters create-auto $CLUSTER \
    --location=$REGION --project=$PROJECT
else
  echo "    Cluster '$CLUSTER' already exists — skipping creation"
fi

echo ""
echo "[1/10] Connecting to cluster..."
gcloud container clusters get-credentials $CLUSTER \
  --location=$REGION --project=$PROJECT

echo ""
echo "[2/10] Creating namespace..."
kubectl apply -f k8s/namespace.yaml

echo ""
echo "[3/10] Installing Strimzi operator (skip if already installed)..."
if ! kubectl get deployment strimzi-cluster-operator -n $NAMESPACE &> /dev/null; then
  kubectl create -f "https://strimzi.io/install/latest?namespace=$NAMESPACE" -n $NAMESPACE
  kubectl wait --for=condition=ready pod \
    -l name=strimzi-cluster-operator -n $NAMESPACE --timeout=120s
else
  echo "   Strimzi operator already installed - skipping"
fi

echo ""
echo "[4/10] Applying ConfigMap..."
kubectl apply -f k8s/configmap.yaml

echo ""
echo "[5/10] Creating secrets from GCP Secret Manager..."
./k8s/create-secrets.sh

echo ""
echo "[6/10] Applying service account..."
kubectl apply -f k8s/service-account.yaml

echo ""
echo "[7/10] Deploying Kafka (KRaft mode)..."
kubectl apply -f k8s/kafka.yaml
echo "    Waiting for Kafka to be ready (up to 5 min)..."
kubectl wait kafka/fraud-kafka --for=condition=Ready --timeout=300s -n $NAMESPACE

echo ""
echo "[8/10] Creating Kafka topics..."
kubectl apply -f k8s/kafka-topics.yaml

echo ""
echo "[9/10] Deploying ChromaDB server..."
kubectl apply -f k8s/chromadb-server.yaml
kubectl wait --for=condition=available deployment/chromadb-server \
  -n $NAMESPACE --timeout=120s

echo ""
echo "Checking if model already trained..."
if kubectl get job model-trainer -n $NAMESPACE &> /dev/null; then
  echo "    model-trainer job already exists — skipping (model already trained)"
  echo "    To retrain: kubectl delete job model-trainer -n $NAMESPACE && ./k8s/deploy.sh"
else
  echo "    Running model-trainer job..."
  kubectl apply -f k8s/model-trainer-job.yaml
  kubectl wait --for=condition=complete job/model-trainer \
    -n $NAMESPACE --timeout=300s
fi

echo ""
echo "[10/10] Deploying application services..."
kubectl apply -f k8s/spring-boot.yaml
kubectl apply -f k8s/python-services.yaml

echo ""
echo "Deploying Kafka UI..."
kubectl apply -f k8s/kafka-ui.yaml

echo ""
echo "=================================================="
echo "  Deployment complete. Checking pod status..."
echo "=================================================="
kubectl get pods -n $NAMESPACE

echo ""
echo "Dashboard and kafka-ui URL (may take 1-2 min for LoadBalancer IP to appear):"
echo "Dashboard URL:      kubectl get service streamlit -n $NAMESPACE"
echo "Kafka UI URL:       kubectl get service kafka-ui -n $NAMESPACE"
echo ""
echo "Run test scenario:  kubectl apply -f k8s/test-data-generator-job.yaml"
echo "Watch logs:         kubectl logs -f deployment/inference-consumer -n $NAMESPACE"
