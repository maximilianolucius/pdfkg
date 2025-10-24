#!/bin/bash
# macOS setup script for PDFKG

set -e  # Exit on error

echo "=========================================="
echo "PDFKG macOS Setup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}ERROR: This script is for macOS only${NC}"
    echo "For Linux, use: docker-compose up -d"
    exit 1
fi

echo -e "${GREEN}✓ Running on macOS${NC}"
echo ""

# Check Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not installed${NC}"
    echo "Install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi
echo -e "${GREEN}✓ Docker found${NC}"

# Check Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not running${NC}"
    echo "Start Docker Desktop and try again"
    exit 1
fi
echo -e "${GREEN}✓ Docker is running${NC}"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 is not installed${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
echo ""

# Step 1: Clean up existing containers
echo "=========================================="
echo "Step 1: Cleaning up existing containers"
echo "=========================================="
if docker ps -a | grep -q "pdfkg-"; then
    echo "Stopping and removing existing containers..."
    docker-compose -f docker-compose.mac.yml down -v 2>/dev/null || true
    echo -e "${GREEN}✓ Cleanup complete${NC}"
else
    echo "No existing containers found"
fi
echo ""

# Step 2: Install Python dependencies
echo "=========================================="
echo "Step 2: Installing Python dependencies"
echo "=========================================="
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}ERROR: requirements.txt not found${NC}"
    exit 1
fi

pip3 install -r requirements.txt

# Fix google-generativeai version
echo ""
echo "Fixing google-generativeai version..."
pip3 uninstall google-generativeai -y 2>/dev/null || true
pip3 install google-generativeai==0.8.5

GENAI_VERSION=$(python3 -c "import google.generativeai as genai; print(genai.__version__)" 2>/dev/null || echo "ERROR")
if [ "$GENAI_VERSION" = "0.8.5" ]; then
    echo -e "${GREEN}✓ google-generativeai 0.8.5 installed${NC}"
else
    echo -e "${YELLOW}⚠ Warning: google-generativeai version is $GENAI_VERSION (expected 0.8.5)${NC}"
fi
echo ""

# Step 3: Check/create .env file
echo "=========================================="
echo "Step 3: Checking environment configuration"
echo "=========================================="
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cat > .env << 'EOF'
# Database connections
ARANGO_HOST=localhost
ARANGO_PORT=8529
ARANGO_USER=root
ARANGO_PASSWORD=
ARANGO_DB=pdfkg

MILVUS_HOST=localhost
MILVUS_PORT=19530

# Storage backend
STORAGE_BACKEND=arango

# Embedding dimension
DEFAULT_EMBED_DIM=384

# LLM APIs (optional - add your keys)
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash

MISTRAL_API_KEY=
MISTRAL_MODEL=mistral-large-latest
EOF
    echo -e "${GREEN}✓ Created .env file${NC}"
    echo -e "${YELLOW}⚠ Please edit .env and add your API keys if needed${NC}"
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi
echo ""

# Step 4: Start Docker containers
echo "=========================================="
echo "Step 4: Starting Docker containers (Mac)"
echo "=========================================="
echo "This may take 30-60 seconds..."

docker-compose -f docker-compose.mac.yml up -d

echo "Waiting for services to initialize..."
sleep 30

# Check if containers are running
if docker ps | grep -q "pdfkg-arangodb" && docker ps | grep -q "pdfkg-milvus"; then
    echo -e "${GREEN}✓ Containers started successfully${NC}"
else
    echo -e "${RED}ERROR: Some containers failed to start${NC}"
    echo "Check logs with: docker-compose -f docker-compose.mac.yml logs"
    exit 1
fi
echo ""

# Step 5: Verify connections
echo "=========================================="
echo "Step 5: Verifying database connections"
echo "=========================================="

# Test ArangoDB
if docker exec pdfkg-arangodb arangosh --server.endpoint tcp://localhost:8529 --javascript.execute-string "db._version()" &> /dev/null; then
    echo -e "${GREEN}✓ ArangoDB is responding${NC}"
else
    echo -e "${YELLOW}⚠ ArangoDB connection test failed (may need more time)${NC}"
fi

# Test Milvus
if curl -s http://localhost:9091/healthz &> /dev/null; then
    echo -e "${GREEN}✓ Milvus is responding${NC}"
else
    echo -e "${YELLOW}⚠ Milvus connection test failed (may need more time)${NC}"
fi
echo ""

# Step 6: Create data directories
echo "=========================================="
echo "Step 6: Creating data directories"
echo "=========================================="
mkdir -p data/input data/output
echo -e "${GREEN}✓ Data directories created${NC}"
echo ""

# Final summary
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep pdfkg || true
echo ""
echo "Next Steps:"
echo "1. Add your PDF files to: data/input/"
echo "2. (Optional) Add API keys to .env file"
echo "3. Start the application: python3 app.py"
echo ""
echo "Useful Commands:"
echo "  View logs:   docker-compose -f docker-compose.mac.yml logs -f"
echo "  Stop:        docker-compose -f docker-compose.mac.yml down"
echo "  Restart:     docker-compose -f docker-compose.mac.yml restart"
echo "  Full reset:  docker-compose -f docker-compose.mac.yml down -v"
echo ""
echo "Troubleshooting:"
echo "  See MAC_SETUP.md for detailed troubleshooting guide"
echo ""
echo -e "${GREEN}Ready to run: python3 app.py${NC}"
