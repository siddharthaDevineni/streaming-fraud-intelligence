#!/bin/bash

echo "Setting up Kafka topics for Streaming-Intelligent Real-time AI agentic fraud detection..."

# Waiting for Kafka to be ready...
echo "Waiting for Kafka to be ready..."
sleep 10

# Function to create topic
create_topic() {
  local topic_name=$1
  local partitions=$2
  local replication=$3

  echo "Creating topic: $topic_name (partitions: $partitions, replication: $replication)"

  docker exec kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic "$topic_name" \
  --partitions "$partitions" \
  --replication-factor "$replication"
}

# Input topics
echo ""
echo "Creating INPUT topics..."
create_topic "transactions" 3 1
create_topic "customerProfiles" 3 1

# Output topics
echo ""
echo "Creating OUTPUT topics..."
create_topic "fraud-alerts" 3 1
create_topic "human-review" 3 1
create_topic "approved-transactions" 3 1

# Feedback Loop topic
echo ""
echo "Creating Feedback topic..."
create_topic "analyst-feedback" 3 1

# Python ML topics
echo ""
echo "Creating Python ML topics..."
create_topic "enriched-transactions" 3 1
create_topic "ml-predictions" 3 1
create_topic "model-health" 3 1
create_topic "fraud-decisions" 3 1

# List all topics
echo ""
echo "Topics created successfully! Current topics:"
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

echo ""
echo "Kafka infrastructure is ready for Streaming-Intelligent Real-time AI Agentic Fraud Detection...!"
echo "Kafka UI available at http://localhost:8090"