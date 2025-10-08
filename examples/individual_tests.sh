#!/bin/bash

# Individual PhotoFlow CLI Test Functions
# Source this file to run individual tests: source examples/individual_tests.sh

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Create output directory
mkdir -p examples/output

# Test 1: Action Discovery
test_action_discovery() {
    echo -e "${BLUE}🔍 Testing Action Discovery${NC}"
    echo "photoflow list-actions"
    photoflow list-actions
    echo
    echo "photoflow list-actions --detailed"
    photoflow list-actions --detailed
    echo
    echo "photoflow info resize"
    photoflow info resize
    echo
}

# Test 2: Single Image Processing
test_single_image() {
    echo -e "${BLUE}📸 Testing Single Image Processing${NC}"
    echo "Basic resize:"
    photoflow process test_images/landscape_4000x3000.jpg --action resize:width=1200 --output examples/output/single_resize.jpg --verbose
    echo
    echo "Multiple actions:"
    photoflow process test_images/portrait_3000x4000.jpg --action resize:width=800 --action watermark:text='Test Watermark' --output examples/output/single_multi.jpg --verbose
    echo
    echo "Dry run:"
    photoflow process test_images/abstract_2000x2000.jpg --action resize:width=600 --dry-run --verbose
    echo
}

# Test 3: Pipeline Operations
test_pipelines() {
    echo -e "${BLUE}🔧 Testing Pipeline Operations${NC}"
    echo "Creating pipeline:"
    photoflow pipeline create test_pipeline --action resize:width=1000 --action watermark:text='Pipeline Test' --output examples/output/test_pipeline.json
    echo
    echo "Listing pipelines:"
    photoflow pipeline list --directory examples/output/
    echo
    echo "Running pipeline:"
    photoflow pipeline run examples/output/test_pipeline.json test_images/landscape_4000x3000.jpg --output examples/output/pipeline_result.jpg --verbose
    echo
}

# Test 4: Batch Processing
test_batch_processing() {
    echo -e "${BLUE}📁 Testing Batch Processing${NC}"
    echo "Simple batch resize:"
    photoflow batch test_images/ --action resize:width=500 --output-dir examples/output/batch_simple --verbose
    echo
    echo "Batch with multiple actions:"
    photoflow batch test_images/ --action resize:width=400 --action watermark:text='Batch Test' --output-dir examples/output/batch_multi --verbose
    echo
    echo "Batch dry run:"
    photoflow batch test_images/ --action resize:width=300 --dry-run
    echo
}

# Test 5: Template System
test_templates() {
    echo -e "${BLUE}📝 Testing Template System${NC}"
    echo "Template in watermark:"
    photoflow process test_images/landscape_4000x3000.jpg --action watermark:text='© <creator> - <filename> (<width>x<height>)' --output examples/output/template_watermark.jpg --verbose
    echo
    echo "Template variables info:"
    photoflow process test_images/portrait_3000x4000.jpg --action watermark:text='<camera_make> <camera_model> - <megapixels>MP' --output examples/output/template_exif.jpg --verbose
    echo
}

# Test 6: Error Handling
test_error_handling() {
    echo -e "${BLUE}⚠️ Testing Error Handling${NC}"
    echo "Invalid action (should fail):"
    photoflow process test_images/landscape_4000x3000.jpg --action invalid_action 2>&1 || echo -e "${GREEN}✅ Handled gracefully${NC}"
    echo
    echo "Invalid parameters (should fail):"
    photoflow process test_images/landscape_4000x3000.jpg --action resize:width=invalid 2>&1 || echo -e "${GREEN}✅ Handled gracefully${NC}"
    echo
    echo "Non-existent file (should fail):"
    photoflow process non_existent.jpg --action resize:width=800 2>&1 || echo -e "${GREEN}✅ Handled gracefully${NC}"
    echo
}

# Test 7: Action Parameter Validation
test_action_validation() {
    echo -e "${BLUE}✅ Testing Action Parameter Validation${NC}"
    echo "Resize with aspect ratio:"
    photoflow process test_images/landscape_4000x3000.jpg --action resize:width=1200,maintain_aspect=true --output examples/output/validation_aspect.jpg --verbose
    echo
    echo "Resize without aspect ratio:"
    photoflow process test_images/landscape_4000x3000.jpg --action resize:width=1200,height=800,maintain_aspect=false --output examples/output/validation_no_aspect.jpg --verbose
    echo
    echo "Crop with coordinates:"
    photoflow process test_images/abstract_2000x2000.jpg --action crop:x=100,y=100,width=1000,height=1000 --output examples/output/validation_crop.jpg --verbose
    echo
    echo "Watermark with position:"
    photoflow process test_images/portrait_3000x4000.jpg --action watermark:text='Corner Test',position=top_left --output examples/output/validation_watermark.jpg --verbose
    echo
}

# Test 8: Performance Testing
test_performance() {
    echo -e "${BLUE}⚡ Testing Performance${NC}"
    # Create test files
    mkdir -p examples/output/perf_test
    for i in {1..3}; do
        cp test_images/landscape_4000x3000.jpg "examples/output/perf_test/test_${i}.jpg"
    done

    echo "Batch processing performance:"
    time photoflow batch examples/output/perf_test/ --action resize:width=800 --output-dir examples/output/perf_results --verbose
    echo
}

# Test 9: Configuration
test_configuration() {
    echo -e "${BLUE}⚙️ Testing Configuration${NC}"
    # Create test config
    cat > examples/output/test_config.yaml << 'EOF'
default_output_format: "PNG"
default_quality: 95
log_level: "DEBUG"
EOF

    echo "Using configuration file:"
    photoflow --config examples/output/test_config.yaml list-actions
    echo
}

# Test 10: Complex Workflows
test_complex_workflows() {
    echo -e "${BLUE}🎯 Testing Complex Workflows${NC}"

    # Create complex pipeline
    cat > examples/output/complex_pipeline.json << 'EOF'
{
  "name": "complex_workflow",
  "description": "Complex image processing workflow",
  "version": "1.0",
  "actions": [
    {
      "name": "resize",
      "parameters": {
        "width": 1600,
        "maintain_aspect": true
      },
      "description": "Resize to HD width"
    },
    {
      "name": "watermark",
      "parameters": {
        "text": "📸 Processed by PhotoFlow - <filename>",
        "position": "bottom_right",
        "opacity": 80
      },
      "description": "Add branding watermark"
    },
    {
      "name": "save",
      "parameters": {
        "format": "WebP",
        "quality": 85,
        "optimize": true
      },
      "description": "Save as optimized WebP"
    }
  ]
}
EOF

    echo "Running complex pipeline:"
    photoflow pipeline run examples/output/complex_pipeline.json test_images/landscape_4000x3000.jpg --output examples/output/complex_result.jpg --verbose
    echo
}

# Summary function
run_all_tests() {
    echo -e "${GREEN}🧪 Running All PhotoFlow CLI Tests${NC}"
    echo "=================================="
    echo

    test_action_discovery
    test_single_image
    test_pipelines
    test_batch_processing
    test_templates
    test_error_handling
    test_action_validation
    test_performance
    test_configuration
    test_complex_workflows

    echo -e "${GREEN}🎉 All tests completed!${NC}"
    echo "Check examples/output/ for all generated files."
}

# Usage information
show_usage() {
    echo "PhotoFlow CLI Individual Test Functions"
    echo "======================================"
    echo
    echo "Usage: source examples/individual_tests.sh"
    echo
    echo "Available test functions:"
    echo "  test_action_discovery    - Test action listing and info"
    echo "  test_single_image        - Test single image processing"
    echo "  test_pipelines           - Test pipeline operations"
    echo "  test_batch_processing    - Test batch processing"
    echo "  test_templates           - Test template system"
    echo "  test_error_handling      - Test error handling"
    echo "  test_action_validation   - Test parameter validation"
    echo "  test_performance         - Test performance"
    echo "  test_configuration       - Test configuration"
    echo "  test_complex_workflows   - Test complex workflows"
    echo "  run_all_tests           - Run all tests"
    echo
    echo "Example:"
    echo "  source examples/individual_tests.sh"
    echo "  test_single_image"
    echo "  test_pipelines"
    echo "  run_all_tests"
}

# If script is run directly (not sourced), show usage
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    show_usage
fi
