#!/bin/bash
# Builds both Docker images and pushes them to Artifact Registry.
# Run this whenever code changes, before ./k8s/deploy.sh

set -e

# Always run from the project root, regardless of where this script is invoked from
cd "$(dirname "$0")/.."

REGISTRY=europe-west3-docker.pkg.dev/streaming-fraud-intelligence/streaming-fraud-intelligence

echo "Building images..."
docker compose build spring-boot inference-consumer

echo "Building testgen image (TestDataGenerator — src/test/java class,"
echo "needs Maven + full source, not bundled into the production app.jar)..."
docker build --platform linux/amd64 -f Dockerfile.testgen \
  -t $REGISTRY/testgen:latest .

echo "Tagging images..."
docker tag agentic-fraud-detection-ai-kafka-spring-boot:latest "$REGISTRY"/spring-boot:latest
docker tag streaming-fraud-intelligence/python-ml:latest "$REGISTRY"/python-ml:latest

echo "Pushing to Artifact Registry..."
docker push "$REGISTRY"/spring-boot:latest
docker push "$REGISTRY"/python-ml:latest

echo "Done. Images pushed:"
echo "  $REGISTRY/spring-boot:latest"
echo "  $REGISTRY/python-ml:latest"
echo "  $REGISTRY/testgen:latest"