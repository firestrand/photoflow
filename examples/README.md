# PhotoFlow CLI Testing Examples

This directory contains comprehensive testing scripts for the PhotoFlow CLI to help you manually verify all features work correctly.

## Quick Start

```bash
# Run the quick test (5 minutes)
./examples/quick_test.sh

# Run comprehensive tests (15 minutes)
./examples/cli_examples.sh
```

## Test Scripts

### 1. `quick_test.sh` - Fast Testing
**Duration:** ~5 minutes
**Purpose:** Quick verification of core features

**Features Tested:**
- Basic CLI commands
- Single image processing
- Pipeline creation/execution
- Batch processing
- Template system

```bash
./examples/quick_test.sh
```

### 2. `cli_examples.sh` - Comprehensive Testing
**Duration:** ~15 minutes
**Purpose:** Complete feature testing with detailed output

**Features Tested:**
- Action discovery and information
- Single image processing (all actions)
- Pipeline management (create, list, run)
- Batch processing with filtering
- Template system with variables
- Error handling and validation
- Performance testing
- Configuration file support
- Edge cases and error conditions

```bash
./examples/cli_examples.sh
```

### 3. `individual_tests.sh` - Modular Testing
**Duration:** Variable (run specific tests)
**Purpose:** Test individual features in isolation

**Usage:**
```bash
# Source the script to load test functions
source examples/individual_tests.sh

# Run specific tests
test_single_image
test_pipelines
test_batch_processing

# Or run all tests
run_all_tests
```

**Available Test Functions:**
- `test_action_discovery` - Action listing and info
- `test_single_image` - Single image processing
- `test_pipelines` - Pipeline operations
- `test_batch_processing` - Batch processing
- `test_templates` - Template system
- `test_error_handling` - Error handling
- `test_action_validation` - Parameter validation
- `test_performance` - Performance testing
- `test_configuration` - Configuration files
- `test_complex_workflows` - Complex workflows

## Expected Outputs

After running the tests, check the `examples/output/` directory for:

### Files Created:
- `resized_landscape.jpg` - Resized landscape image
- `processed_portrait.jpg` - Portrait with resize + watermark
- `cropped_abstract.jpg` - Cropped abstract image
- `social_media_result.jpg` - Pipeline processing result
- `template_test.jpg` - Template variable demonstration
- `social_media.json` - Social media pipeline file
- `thumbnails.json` - Thumbnail pipeline file

### Directories Created:
- `batch_resized/` - Batch processing results
- `batch_thumbnails/` - Batch thumbnail results
- `batch_filtered/` - Filtered batch results
- `performance_results/` - Performance test results

## Verification Checklist

### ✅ CLI Commands
- [ ] `photoflow --version` shows version
- [ ] `photoflow --help` shows help text
- [ ] `photoflow list-actions` lists 4 actions
- [ ] `photoflow info resize` shows detailed info

### ✅ Single Image Processing
- [ ] Resize works with aspect ratio preservation
- [ ] Multiple actions can be chained
- [ ] Watermark templates are evaluated
- [ ] Crop coordinates work correctly
- [ ] Dry run shows plan without processing

### ✅ Pipeline Management
- [ ] Pipeline creation works
- [ ] Pipeline listing works
- [ ] Pipeline execution works
- [ ] Template variables in pipelines work

### ✅ Batch Processing
- [ ] Directory processing works
- [ ] Progress bars display
- [ ] File pattern filtering works
- [ ] Output directory creation works
- [ ] Error handling for empty directories

### ✅ Template System
- [ ] Basic variables work (`<width>`, `<height>`)
- [ ] EXIF variables work (`<creator>`, `<camera_make>`)
- [ ] Computed variables work (`<megapixels>`)
- [ ] Template functions work (`upper()`, etc.)

### ✅ Error Handling
- [ ] Invalid actions show helpful errors
- [ ] Invalid parameters show validation errors
- [ ] Non-existent files show file errors
- [ ] Empty directories handled gracefully

### ✅ Performance
- [ ] Batch processing completes
- [ ] Memory usage reasonable
- [ ] Progress tracking works
- [ ] Large files handled correctly

## Troubleshooting

### Common Issues:

1. **CLI not found:**
   ```bash
   pip install -e .
   ```

2. **Test images missing:**
   ```bash
   # The test scripts expect test_images/ directory
   # Make sure it exists with sample images
   ```

3. **Permission errors:**
   ```bash
   chmod +x examples/*.sh
   ```

4. **Output directory issues:**
   ```bash
   # Scripts automatically create examples/output/
   # Make sure you have write permissions
   ```

## Manual Testing Tips

1. **Start with quick test:** Run `quick_test.sh` first to verify basic functionality
2. **Check outputs:** Always verify files are created in `examples/output/`
3. **Test edge cases:** Try invalid inputs to test error handling
4. **Performance testing:** Use larger images for performance validation
5. **Template testing:** Create custom templates to test the template system

## Expected Results

All tests should complete successfully with:
- No fatal errors or crashes
- All expected output files created
- Proper error handling for invalid inputs
- Clear, helpful error messages
- Progress indicators working
- Template variables properly evaluated

If any tests fail, check the error output and verify your PhotoFlow installation is correct.
