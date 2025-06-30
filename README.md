# PhotoFlow

A powerful Python library for batch photo and image manipulation with visual workflow design, AI-assisted processing, and enterprise-grade features.

## Features

### Core Processing
- **Batch Processing**: Process thousands of images via reusable action lists
- **Visual Node Editor**: Compose and tune pipelines with drag-and-drop workflows
- **Live Preview**: Real-time preview of individual actions and complete pipelines
- **Cross-Platform GUI**: Native look-and-feel built with Dear PyGui

### Processing Modes
- **Command-Line Interface**: Full feature parity for scripted and headless operation
- **Droplet Mode**: Drag-and-drop files for instant processing
- **Watch Folders**: Automatically process new images as they arrive
- **Worker/Queue Mode**: Headless processing for render farms and scheduled jobs

### Image Operations
- **Comprehensive Actions**: 50+ built-in actions including resize, rotate, effects, watermarks
- **AI-Assisted Processing**: Auto-enhance, background removal, upscaling with open-source models
- **Metadata Management**: Full EXIF and IPTC support with editing capabilities
- **Format Support**: Modern formats (RAW, HEIF, AVIF) via pluggable readers

### Enterprise Features
- **REST API**: Full programmatic access for CI/CD integration
- **Cloud Integration**: Direct S3, GCS, and Azure Blob storage support
- **GPU Acceleration**: CUDA and OpenCL support for high-performance processing
- **Plugin SDK**: Extensible architecture for custom actions
- **Safe Mode**: Sandboxed execution prevents untrusted code execution

### Quality & Reliability
- **Duplicate Detection**: Identify and flag redundant images before processing
- **Versioned Pipelines**: Rollback support for reproducible workflows
- **Error Recovery**: Robust logging and recovery mechanisms
- **Performance Monitoring**: Built-in metrics and optimization tools

## Quick Start

### Installation

```bash
# Install from PyPI (when available)
pip install photoflow

# Or install from source
git clone https://github.com/firestrand/photoflow.git
cd photoflow
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Process a single image
photoflow resize --width 800 input.jpg output.jpg

# Apply action list to multiple images
photoflow batch --actions resize_web.actions *.jpg

# Watch folder for new images
photoflow watch --input ./incoming --actions process.actions --output ./processed

# Start GUI
photoflow gui
```

### Action Lists

Create reusable `.actions` files (JSON format) for consistent processing:

```json
[
  {
    "action": "resize",
    "width": 1920,
    "height": 1080,
    "maintain_aspect": true
  },
  {
    "action": "watermark",
    "text": "© 2025 Your Company",
    "position": "bottom_right",
    "opacity": 0.7
  },
  {
    "action": "save",
    "format": "webp",
    "quality": 85
  }
]
```

## Architecture

PhotoFlow is built on SOLID principles with a modular, extensible architecture enhanced by proven patterns from mature image processing tools:

- **Protocol-Based Design**: All components implement clear interfaces
- **Dependency Injection**: Fully testable and configurable
- **Form-Field Architecture**: Specialized parameter types with built-in validation
- **Template Expression System**: Safe dynamic evaluation for file naming and parameters
- **Rich Metadata Context**: Structured EXIF/IPTC data with template variables
- **Dynamic Field Relevance**: Contextual UIs showing only relevant parameters
- **Processing Mixins**: Reusable operation patterns for common tasks
- **Immutable Models**: Predictable data flow and processing

## Configuration

Configure PhotoFlow via YAML files or environment variables:

```yaml
# photoflow.yaml
max_workers: 8
gpu_acceleration: true
default_quality: 95
theme: dark
preview_size: 512

plugin_dirs:
  - ~/.photoflow/plugins
  - ./custom_actions
```

Environment variables use the `PHOTOFLOW_` prefix:
```bash
export PHOTOFLOW_MAX_WORKERS=16
export PHOTOFLOW_GPU_ACCELERATION=true
```

## Development

PhotoFlow follows test-driven development with comprehensive quality checks:

```bash
# Run tests
pytest

# Check code quality
ruff check photoflow/ tests/
python -m black photoflow/ tests/

# Run all checks
pre-commit run --all-files
```

### Project Structure

```
photoflow/
├── photoflow/          # Main package
│   ├── core/          # Domain models and protocols
│   ├── actions/       # Image processing actions
│   ├── cli/           # Command-line interface
│   └── gui/           # Graphical interface
├── tests/             # Test suite
│   ├── unit/          # Unit tests
│   ├── integration/   # Integration tests
│   └── e2e/           # End-to-end tests
└── dev-doc/           # Development documentation
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and coverage remains above 90%
5. Submit a pull request

## Status

🚧 **In Development** - Phase 1.2 Complete

- ✅ Core architecture and foundation
- ✅ Settings and configuration system
- ✅ Testing framework and quality assurance
- ✅ Immutable domain models with comprehensive validation
- ✅ JSON/YAML serialization system
- 🔄 **Next**: Field-based parameter system and template evaluation (Phase 1.3)
- 📋 Form-based action system (Phase 1.4)
- 📋 CLI with rich output and validation (Phase 2)
- 📋 GUI with dynamic forms (Phase 4+)
