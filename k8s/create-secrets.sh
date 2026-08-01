#!/bin/bash
# Creates K8s secret from GCP Secret Manager values
# Run once after cluster connection is established
# Never commit actual secret values to git

set -e

NAMESPACE=fraud-detection
PROJECT=streaming-fraud-intelligence

echo "Fetching secrets from GCP Secret Manager...."

GROQ_KEY=$(gcloud secrets versions access latest \
  --secret=groq-api-key --project="$PROJECT")

LANGCHAIN_KEY=$(gcloud secrets versions access latest \
--secret=langchain-api-key --project="$PROJECT")

echo "Creating K8s secret..."

kubectl create secret generic app-secrets \
  --namespace="$NAMESPACE" --from-literal=groq-api-key="$GROQ_KEY" \
  --from-literal=langchain-api-key="$LANGCHAIN_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Secret created successfully"
kubectl get secret app-secrets -n "$NAMESPACE"

