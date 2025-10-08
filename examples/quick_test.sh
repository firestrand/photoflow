#!/bin/bash

# Quick PhotoFlow CLI Test Script
# A condensed version for rapid testing of core features

echo "🚀 Quick PhotoFlow CLI Test"
echo "========================="
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Helper function
test_cmd() {
    echo -e "${YELLOW}$ $1${NC}"
    eval $1
    echo
}

# Check installation
if ! command -v photoflow &> /dev/null; then
    echo "❌ PhotoFlow CLI not found. Install with: pip install -e ."
    exit 1
fi

# Create output directory
mkdir -p examples/output

echo "1. Basic Commands"
test_cmd "photoflow --version"
test_cmd "photoflow list-actions"

echo "2. Single Image Processing"
test_cmd "photoflow process test_images/landscape_4000x3000.jpg --action resize:width=1200 --output examples/output/quick_test.jpg --verbose"

echo "3. Pipeline Creation & Execution"
test_cmd "photoflow pipeline create quick_demo --action resize:width=800 --action watermark:text='Quick Test' --output examples/output/quick_demo.json"
test_cmd "photoflow pipeline run examples/output/quick_demo.json test_images/portrait_3000x4000.jpg --output examples/output/quick_pipeline_result.jpg --verbose"

echo "4. Batch Processing"
test_cmd "photoflow batch test_images/ --action resize:width=600 --output-dir examples/output/quick_batch --verbose"

echo "5. Template System"
test_cmd "photoflow process test_images/abstract_2000x2000.jpg --action watermark:text='© <filename> - <width>x<height>px' --output examples/output/template_demo.jpg --verbose"

echo -e "${GREEN}✅ Quick test complete!${NC}"
echo "Check examples/output/ for generated files"
