#!/bin/bash

# PhotoFlow CLI Demo Showcase
# A demonstration script showing the key features of PhotoFlow CLI

echo "🎨 PhotoFlow CLI Demo Showcase"
echo "============================="
echo "This demo showcases the key features of PhotoFlow CLI"
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Create output directory
mkdir -p examples/output

echo -e "${BLUE}📋 What PhotoFlow CLI Can Do:${NC}"
echo "• Process single images with multiple actions"
echo "• Create and manage reusable pipelines"
echo "• Batch process entire directories"
echo "• Use templates for dynamic text and naming"
echo "• Rich progress tracking and error handling"
echo

echo -e "${BLUE}🎯 Available Actions:${NC}"
photoflow list-actions
echo

echo -e "${BLUE}📸 Demo 1: Single Image Processing${NC}"
echo "Processing landscape image with resize and watermark..."
photoflow process test_images/landscape_4000x3000.jpg \
  --action resize:width=1600 \
  --action watermark:text='© PhotoFlow Demo - <filename>' \
  --output examples/output/demo_landscape.jpg \
  --verbose
echo

echo -e "${BLUE}🔧 Demo 2: Pipeline Creation${NC}"
echo "Creating a social media optimization pipeline..."
photoflow pipeline create social_media_optimizer \
  --action resize:width=1200 \
  --action watermark:text='📸 <filename> - <width>x<height>px' \
  --output examples/output/social_media_optimizer.json \
  --description "Optimize images for social media posting"
echo

echo -e "${BLUE}⚙️ Demo 3: Pipeline Execution${NC}"
echo "Running the social media pipeline on a portrait..."
photoflow pipeline run examples/output/social_media_optimizer.json \
  test_images/portrait_3000x4000.jpg \
  --output examples/output/social_media_portrait.jpg \
  --verbose
echo

echo -e "${BLUE}📁 Demo 4: Batch Processing${NC}"
echo "Batch processing all images to create thumbnails..."
photoflow batch test_images/ \
  --action resize:width=400 \
  --action watermark:text='Thumbnail - <filename>' \
  --output-dir examples/output/thumbnails \
  --verbose
echo

echo -e "${BLUE}📝 Demo 5: Template System${NC}"
echo "Demonstrating template variables with detailed watermark..."
photoflow process test_images/abstract_2000x2000.jpg \
  --action watermark:text='<format> Image - <width>x<height> - <megapixels>MP' \
  --output examples/output/template_showcase.jpg \
  --verbose
echo

echo -e "${BLUE}🔍 Demo 6: Action Information${NC}"
echo "Getting detailed information about the resize action..."
photoflow info resize
echo

echo -e "${BLUE}⚡ Demo 7: Performance Test${NC}"
echo "Testing batch processing performance..."
# Create test files
mkdir -p examples/output/perf_test
for i in {1..3}; do
    cp test_images/landscape_4000x3000.jpg "examples/output/perf_test/image_${i}.jpg"
done

echo "Processing 3 images with performance timing..."
time photoflow batch examples/output/perf_test/ \
  --action resize:width=800 \
  --action watermark:text='Batch <filename>' \
  --output-dir examples/output/perf_results \
  --verbose
echo

echo -e "${BLUE}🎯 Demo 8: Complex Pipeline${NC}"
echo "Creating and running a complex multi-step pipeline..."

# Create complex pipeline file
cat > examples/output/complex_demo.json << 'EOF'
{
  "name": "complex_processing",
  "description": "Complex multi-step image processing pipeline",
  "version": "1.0",
  "actions": [
    {
      "name": "resize",
      "parameters": {
        "width": 1920,
        "maintain_aspect": true
      },
      "description": "Resize to Full HD width"
    },
    {
      "name": "watermark",
      "parameters": {
        "text": "🎨 PhotoFlow Complex Demo - <filename>",
        "position": "bottom_right",
        "opacity": 85
      },
      "description": "Add branded watermark"
    },
    {
      "name": "save",
      "parameters": {
        "format": "PNG",
        "quality": 95,
        "optimize": true
      },
      "description": "Save as optimized PNG"
    }
  ]
}
EOF

echo "Running complex pipeline:"
photoflow pipeline run examples/output/complex_demo.json \
  test_images/landscape_4000x3000.jpg \
  --output examples/output/complex_result.jpg \
  --verbose
echo

echo -e "${BLUE}✅ Demo Summary${NC}"
echo "==============="
echo "Generated files in examples/output/:"
echo "• demo_landscape.jpg - Single image processing"
echo "• social_media_portrait.jpg - Pipeline processing"
echo "• thumbnails/ - Batch processing results"
echo "• template_showcase.jpg - Template demonstration"
echo "• complex_result.jpg - Complex pipeline result"
echo "• social_media_optimizer.json - Created pipeline"
echo "• complex_demo.json - Complex pipeline definition"
echo

echo -e "${GREEN}🎉 PhotoFlow CLI Demo Complete!${NC}"
echo
echo "Key Features Demonstrated:"
echo "• ✅ Single image processing with multiple actions"
echo "• ✅ Pipeline creation and management"
echo "• ✅ Batch processing with progress tracking"
echo "• ✅ Template system with dynamic variables"
echo "• ✅ Rich CLI output with colors and progress bars"
echo "• ✅ Error handling and validation"
echo "• ✅ Performance testing"
echo "• ✅ Complex multi-step workflows"
echo
echo "PhotoFlow CLI is ready for production use!"
echo "Run 'photoflow --help' to see all available commands."
