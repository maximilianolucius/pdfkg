#!/bin/bash
# Helper script to start app.py cleanly

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "PDFKG App Starter"
echo "=========================================="
echo ""

# Function to kill existing processes
kill_existing() {
    echo "Checking for existing app.py processes..."
    PIDS=$(pgrep -f "python.*app.py" || true)

    if [ -n "$PIDS" ]; then
        echo -e "${YELLOW}Found existing processes: $PIDS${NC}"
        echo "Killing existing processes..."
        pkill -f "python.*app.py" || true
        sleep 2
        echo -e "${GREEN}✓ Cleaned up old processes${NC}"
    else
        echo "✓ No existing processes found"
    fi
    echo ""
}

# Function to check databases
check_databases() {
    echo "Checking database connections..."

    # Check ArangoDB
    if curl -s http://localhost:8529/_api/version > /dev/null 2>&1; then
        echo -e "${GREEN}✓ ArangoDB is responding${NC}"
    else
        echo -e "${RED}✗ ArangoDB is not responding${NC}"
        echo "  Start with: docker-compose up -d arangodb"
        return 1
    fi

    # Check Milvus
    if curl -s http://localhost:9091/healthz > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Milvus is responding${NC}"
    else
        echo -e "${RED}✗ Milvus is not responding${NC}"
        echo "  Start with: docker-compose up -d milvus"
        return 1
    fi

    echo ""
    return 0
}

# Function to find available port
find_port() {
    for port in 8016 8017 8018 8019 7860; do
        if ! lsof -i :$port > /dev/null 2>&1; then
            echo $port
            return 0
        fi
    done
    return 1
}

# Main execution
echo "Step 1: Clean up existing processes"
kill_existing

echo "Step 2: Verify databases"
if ! check_databases; then
    echo -e "${YELLOW}Starting databases...${NC}"
    docker-compose up -d
    echo "Waiting for databases to start (30 seconds)..."
    sleep 30
    if ! check_databases; then
        echo -e "${RED}ERROR: Databases failed to start${NC}"
        echo "Check logs with: docker-compose logs"
        exit 1
    fi
fi

echo "Step 3: Finding available port"
PORT=$(find_port)
if [ -z "$PORT" ]; then
    echo -e "${RED}ERROR: No available ports found${NC}"
    echo "Ports checked: 8016, 8017, 8018, 8019, 7860"
    exit 1
fi
echo -e "${GREEN}✓ Will use port $PORT${NC}"
echo ""

echo "Step 4: Starting app.py"
echo "=========================================="
echo ""

# Start app in foreground
python app.py

# Note: The script will block here while app runs
# Press Ctrl+C to stop the app
