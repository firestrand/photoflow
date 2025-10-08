#!/bin/bash

# PhotoFlow CLI Examples and Testing Script
# This script demonstrates all CLI features and can be used for manual testing

echo "🎨 PhotoFlow CLI Examples and Testing Script"
echo "==========================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper function for section headers
section() {
    echo -e "${BLUE}📋 $1${NC}"
    echo "---"
}

# Helper function for commands
run_command() {
    echo -e "${YELLOW}$ $1${NC}"
    eval $1
    echo
}

# Check if PhotoFlow is installed
if ! command -v photoflow &> /dev/null; then
    echo -e "${RED}❌ PhotoFlow CLI not found. Please install with: pip install -e .${NC}"
    exit 1
fi

# Create example output directory
mkdir -p examples/output

section "1. Basic CLI Information"
run_command "photoflow --version"
run_command "photoflow --help"

section "2. Action Discovery"
run_command "photoflow list-actions"
run_command "photoflow list-actions --detailed"
run_command "photoflow list-actions --tag geometry"
run_command "photoflow info resize"
run_command "photoflow info watermark"

section "3. Single Image Processing"
echo "Testing single image processing with various actions..."

# Test resize action
run_command "photoflow process test_images/landscape_4000x3000.jpg --action resize:width=1200 --output examples/output/resized_landscape.jpg --verbose"

# Test multiple actions
run_command "photoflow process test_images/portrait_3000x4000.jpg --action resize:width=800 --action watermark:text='© PhotoFlow Demo' --output examples/output/processed_portrait.jpg --verbose"

# Test crop action
run_command "photoflow process test_images/abstract_2000x2000.jpg --action crop:x=200,y=200,width=1200,height=1200 --output examples/output/cropped_abstract.jpg --verbose"

# Test dry run
run_command "photoflow process test_images/landscape_4000x3000.jpg --action resize:width=800 --action watermark:text='Test' --dry-run --verbose"

section "4. Pipeline Management"
echo "Testing pipeline creation and execution..."

# Create a social media pipeline
run_command "photoflow pipeline create social_media --action resize:width=1200 --action watermark:text='© <creator>' --output examples/output/social_media.json --description 'Social media optimization pipeline'"

# Create a thumbnail pipeline
run_command "photoflow pipeline create thumbnails --action resize:width=300 --action crop:x=0,y=0,width=300,height=300 --output examples/output/thumbnails.json --description 'Square thumbnail generation'"

# List pipelines
run_command "photoflow pipeline list --directory examples/output/"

# Run pipeline on single image
run_command "photoflow pipeline run examples/output/social_media.json test_images/landscape_4000x3000.jpg --output examples/output/social_media_result.jpg --verbose"

section "5. Batch Processing"
echo "Testing batch processing features..."

# Simple batch resize
run_command "photoflow batch test_images/ --action resize:width=600 --output-dir examples/output/batch_resized --verbose"

# Batch with pipeline
run_command "photoflow batch test_images/ --pipeline examples/output/thumbnails.json --output-dir examples/output/batch_thumbnails --verbose"

# Batch with pattern filtering
run_command "photoflow batch test_images/ --action resize:width=400 --pattern '*.jpg' --output-dir examples/output/batch_filtered --verbose"

# Batch dry run
run_command "photoflow batch test_images/ --action resize:width=800 --action watermark:text='Batch Test' --dry-run"

section "6. Advanced Features Testing"
echo "Testing advanced CLI features..."

# Test with verbose global flag
run_command "photoflow --verbose process test_images/portrait_3000x4000.jpg --action resize:width=500"

# Test error handling with invalid action
echo -e "${YELLOW}$ photoflow process test_images/landscape_4000x3000.jpg --action invalid_action${NC}"
photoflow process test_images/landscape_4000x3000.jpg --action invalid_action 2>&1 || echo -e "${GREEN}✅ Error handling working correctly${NC}"
echo

# Test error handling with invalid parameters
echo -e "${YELLOW}$ photoflow process test_images/landscape_4000x3000.jpg --action resize:width=invalid${NC}"
photoflow process test_images/landscape_4000x3000.jpg --action resize:width=invalid 2>&1 || echo -e "${GREEN}✅ Parameter validation working correctly${NC}"
echo

section "7. Template System Testing"
echo "Testing template evaluation in actions..."

# Test template variables in watermark
run_command "photoflow process test_images/landscape_4000x3000.jpg --action watermark:text='© <creator> - <width>x<height>' --output examples/output/template_test.jpg --verbose"

# Test template in pipeline
cat > examples/output/template_pipeline.json << 'EOF'
{
  "name": "template_test",
  "description": "Pipeline with template variables",
  "version": "1.0",
  "actions": [
    {
      "name": "resize",
      "parameters": {
        "width": 1000,
        "maintain_aspect": true
      },
      "description": "Resize to 1000px width"
    },
    {
      "name": "watermark",
      "parameters": {
        "text": "📸 <filename> - <megapixels>MP by <creator>",
        "position": "bottom_right",
        "opacity": 75
      },
      "description": "Add detailed watermark"
    }
  ]
}
EOF

run_command "photoflow pipeline run examples/output/template_pipeline.json test_images/portrait_3000x4000.jpg --output examples/output/template_pipeline_result.jpg --verbose"

section "8. Performance Testing"
echo "Testing performance with larger operations..."

# Create multiple copies for performance testing
mkdir -p examples/output/performance_test
for i in {1..5}; do
    cp test_images/landscape_4000x3000.jpg "examples/output/performance_test/test_image_${i}.jpg"
done

# Time batch processing
echo -e "${YELLOW}$ time photoflow batch examples/output/performance_test/ --action resize:width=800 --output-dir examples/output/performance_results --verbose${NC}"
time photoflow batch examples/output/performance_test/ --action resize:width=800 --output-dir examples/output/performance_results --verbose
echo

section "9. Configuration Testing"
echo "Testing configuration and settings..."

# Test with config file (create a sample config)
cat > examples/output/test_config.yaml << 'EOF'
# PhotoFlow test configuration
default_output_format: "PNG"
default_quality: 95
log_level: "INFO"
EOF

run_command "photoflow --config examples/output/test_config.yaml list-actions"

section "10. Edge Cases and Error Handling"
echo "Testing edge cases and error handling..."

# Test with non-existent file
echo -e "${YELLOW}$ photoflow process non_existent.jpg --action resize:width=800${NC}"
photoflow process non_existent.jpg --action resize:width=800 2>&1 || echo -e "${GREEN}✅ Non-existent file handling working${NC}"
echo

# Test with invalid pipeline
echo -e "${YELLOW}$ photoflow pipeline run non_existent.json test_images/landscape_4000x3000.jpg${NC}"
photoflow pipeline run non_existent.json test_images/landscape_4000x3000.jpg 2>&1 || echo -e "${GREEN}✅ Invalid pipeline handling working${NC}"
echo

# Test with empty directory
mkdir -p examples/output/empty_dir
echo -e "${YELLOW}$ photoflow batch examples/output/empty_dir/ --action resize:width=800${NC}"
photoflow batch examples/output/empty_dir/ --action resize:width=800 2>&1 || echo -e "${GREEN}✅ Empty directory handling working${NC}"
echo

section "11. Output Verification"
echo "Verifying outputs were created..."

# Check if files were created
outputs=(
    "examples/output/resized_landscape.jpg"
    "examples/output/processed_portrait.jpg"
    "examples/output/cropped_abstract.jpg"
    "examples/output/social_media_result.jpg"
    "examples/output/template_test.jpg"
    "examples/output/template_pipeline_result.jpg"
    "examples/output/social_media.json"
    "examples/output/thumbnails.json"
)

for output in "${outputs[@]}"; do
    if [ -f "$output" ]; then
        echo -e "${GREEN}✅ $output created successfully${NC}"
    else
        echo -e "${RED}❌ $output not found${NC}"
    fi
done

# Check directories
dirs=(
    "examples/output/batch_resized"
    "examples/output/batch_thumbnails"
    "examples/output/batch_filtered"
    "examples/output/performance_results"
)

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ] && [ "$(ls -A $dir)" ]; then
        count=$(ls -1 "$dir" | wc -l)
        echo -e "${GREEN}✅ $dir created with $count files${NC}"
    else
        echo -e "${RED}❌ $dir not found or empty${NC}"
    fi
done

echo
echo -e "${GREEN}🎉 CLI testing complete!${NC}"
echo "Check the examples/output/ directory for all generated files and pipelines."
echo
echo "Summary of features tested:"
echo "• Action discovery and information"
echo "• Single image processing with multiple actions"
echo "• Pipeline creation, management, and execution"
echo "• Batch processing with progress tracking"
echo "• Template system for dynamic parameters"
echo "• Error handling and validation"
echo "• Performance with multiple files"
echo "• Configuration file support"
echo "• Edge cases and error conditions"
