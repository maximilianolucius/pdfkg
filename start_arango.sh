#!/bin/bash
# Quick script to start ArangoDB using Docker Compose

echo "Starting ArangoDB..."
docker-compose up -d

echo ""
echo "Waiting for ArangoDB to be ready..."
sleep 5

echo ""
echo "✅ ArangoDB is running!"
echo ""
echo "Web UI: http://localhost:8529"
echo "Username: root"
echo "Password: (empty)"
echo ""
echo "To stop: docker-compose down"
echo "To view logs: docker-compose logs -f arangodb"
