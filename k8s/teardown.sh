#!/bin/bash
# Deletes all application resources in the fraud-detection namespace.
# For full cleanup (delete cluster too - stops all GKE billing):
# gcloud container clusters delete streaming-fraud-intel --location=europe-west3

set -e

# Always run from the project root, regardless of where this script is invoked from
cd "$(dirname "$0")/.."

PROJECT=streaming-fraud-intelligence
REGION=europe-west3
NAMESPACE=fraud-detection
CLUSTER=streaming-fraud-intel

echo "Deleting all resources in namespace $NAMESPACE...."
kubectl delete namespace $NAMESPACE --ignore-not-found

echo "Done. Cluster itself is still running (still incurs minimal cost)."
echo  "To fully stop billing, also run:"
echo "  gcloud container clusters delete $CLUSTER --location=$REGION --project=$PROJECT"