#!/bin/bash
# PhotoFlow Metadata Management Demo

echo "🔍 PhotoFlow Metadata Management Demo"
echo "====================================="
echo

# Check if metadata dependencies are available
if ! python -c "import piexif, iptcinfo3" 2>/dev/null; then
    echo "❌ Metadata dependencies not installed."
    echo "   Install with: pip install piexif iptcinfo3"
    echo "   Or: pip install -e .[metadata]"
    exit 1
fi

echo "📋 Available metadata actions:"
python -m photoflow.cli.main list-actions | grep metadata
echo

echo "🔍 1. Inspect metadata in original image"
echo "----------------------------------------"
python -m photoflow.cli.main metadata inspect test_images/landscape_4000x3000.jpg
echo

echo "✏️ 2. Write metadata using action system"
echo "----------------------------------------"
python -m photoflow.cli.main process test_images/landscape_4000x3000.jpg \
    --action "write_metadata:creator=John Smith,title=Mountain Vista,keywords=landscape,nature,mountains" \
    --output test_images_processed/landscape_with_metadata.jpg
echo

echo "🔍 3. Inspect metadata in processed image"
echo "----------------------------------------"
python -m photoflow.cli.main metadata inspect test_images_processed/landscape_with_metadata.jpg
echo

echo "📤 4. Extract metadata to JSON file"
echo "-----------------------------------"
python -m photoflow.cli.main process test_images_processed/landscape_with_metadata.jpg \
    --action "extract_metadata:output_format=json" \
    --output test_images_processed/landscape_extracted.jpg
echo

echo "📋 5. View extracted metadata file"
echo "---------------------------------"
if [ -f "test_images_processed/landscape_with_metadata_metadata.json" ]; then
    echo "Contents of metadata file:"
    cat test_images_processed/landscape_with_metadata_metadata.json | head -20
else
    echo "Metadata file not found"
fi
echo

echo "🧹 6. Remove personal metadata"
echo "------------------------------"
python -m photoflow.cli.main process test_images_processed/landscape_with_metadata.jpg \
    --action "remove_metadata:metadata_type=personal" \
    --output test_images_processed/landscape_privacy_clean.jpg
echo

echo "🔍 7. Inspect cleaned image"
echo "---------------------------"
python -m photoflow.cli.main metadata inspect test_images_processed/landscape_privacy_clean.jpg
echo

echo "🎯 Demo completed!"
echo "Created files:"
echo "• test_images_processed/landscape_with_metadata.jpg - Image with metadata"
echo "• test_images_processed/landscape_with_metadata_metadata.json - Extracted metadata"
echo "• test_images_processed/landscape_privacy_clean.jpg - Privacy-cleaned image"
