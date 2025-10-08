#!/usr/bin/env python3
"""
PhotoFlow End-to-End Demonstration

This script demonstrates the complete PhotoFlow workflow from image generation
through pipeline processing, showcasing all currently implemented features.
"""

import tempfile
from pathlib import Path
from typing import Any, ClassVar

# Import PhotoFlow components
from photoflow.core.actions import ActionRegistry, BaseAction
from photoflow.core.fields import BooleanField, IntegerField, StringField
from photoflow.core.mixins import GeometryMixin, ProcessingMixin
from photoflow.core.models import ActionSpec, ImageAsset, Pipeline
from photoflow.core.serialization import load_pipeline, save_pipeline
from photoflow.core.templates import TemplateEvaluator
from tests.utils.test_image_generator import TestImageGenerator

# Try to load images if possible
try:
    from photoflow.core.image_loader import load_image_asset

    IMAGE_LOADING_AVAILABLE = True
except ImportError:
    IMAGE_LOADING_AVAILABLE = False


class DemoResizeAction(BaseAction, GeometryMixin):
    """Demo resize action."""

    name = "resize"
    label = "Resize Image"
    description = "Resize image to specified dimensions"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["geometry", "basic"]

    def setup_fields(self) -> None:
        self.fields = {
            "width": IntegerField(default=1920, min_value=1, max_value=10000),
            "maintain_aspect": BooleanField(default=True),
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        width = params["width"]
        maintain_aspect = params.get("maintain_aspect", True)

        if maintain_aspect:
            aspect_ratio = image.width / image.height
            height = int(width / aspect_ratio)
        else:
            height = image.height

        return image.with_updates(
            size=(width, height),
            processing_metadata={
                "last_action": self.name,
                "processing_history": [
                    *image.processing_metadata.get("processing_history", []),
                    f"resize({width}x{height})",
                ],
            },
        )


class DemoWatermarkAction(BaseAction, ProcessingMixin):
    """Demo watermark action with template evaluation."""

    name = "watermark"
    label = "Add Watermark"
    description = "Add text watermark with template support"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["text", "branding"]

    def setup_fields(self) -> None:
        self.fields = {
            "text": StringField(
                default="© <creator>",
                label="Watermark Text",
                help_text="Supports templates like <creator>, <title>, etc.",
            ),
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        text_template = params["text"]

        # Use template engine
        evaluator = TemplateEvaluator()
        evaluated_text = evaluator.evaluate(text_template, image.template_vars)

        return image.with_updates(
            metadata={"watermark_text": evaluated_text},
            processing_metadata={
                "last_action": self.name,
                "processing_history": [
                    *image.processing_metadata.get("processing_history", []),
                    f"watermark({evaluated_text})",
                ],
            },
        )


def demonstrate_photoflow() -> None:  # noqa: PLR0915 - demo flow intentionally verbose
    """Run the complete PhotoFlow demonstration."""

    print("🎨 PhotoFlow End-to-End Demonstration")
    print("=" * 50)

    # Step 1: Generate test image
    print("\n📸 Step 1: Generating test image...")
    try:
        generator = TestImageGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = generator.create_test_image(
                width=3000,
                height=2000,
                output_path=Path(tmpdir) / "demo_landscape.jpg",
                image_type="landscape",
                camera_make="Canon",
                camera_model="EOS R5",
                photographer="Demo Photographer",
                title="Beautiful Landscape",
                keywords=["landscape", "nature", "photography"],
            )

            print(f"  ✅ Generated: {image_path}")
            print(f"  📏 Size: {image_path.stat().st_size / 1024:.1f} KB")

            # Step 2: Load image
            print("\n📖 Step 2: Loading image with metadata...")
            if IMAGE_LOADING_AVAILABLE:
                image_asset = load_image_asset(image_path)
                # Add IPTC manually for demo
                image_asset = image_asset.with_iptc_update(
                    {
                        "ObjectName": "Beautiful Landscape",
                        "By-line": "Demo Photographer",
                        "Keywords": ["landscape", "nature", "photography"],
                    }
                )
                print(f"  ✅ Loaded: {image_asset.width}x{image_asset.height} {image_asset.format}")
                print(f"  🏷️  EXIF keys: {list(image_asset.exif.keys())[:5]}")
            else:
                # Fallback manual creation
                image_asset = ImageAsset(
                    path=image_path,
                    format="JPEG",
                    size=(3000, 2000),
                    mode="RGB",
                    exif={"Make": "Canon", "Model": "EOS R5"},
                    iptc={"By-line": "Demo Photographer", "ObjectName": "Beautiful Landscape"},
                )
                print(f"  ✅ Created manually: {image_asset.width}x{image_asset.height}")

            # Step 3: Set up action pipeline
            print("\n⚙️  Step 3: Setting up processing pipeline...")
            registry = ActionRegistry()
            registry.register(DemoResizeAction)
            registry.register(DemoWatermarkAction)

            pipeline = Pipeline(
                name="social_media_workflow",
                actions=(
                    ActionSpec(name="resize", parameters={"width": 1200, "maintain_aspect": True}),
                    ActionSpec(name="watermark", parameters={"text": "📷 @<creator> - <title>"}),
                ),
                metadata={"target": "social_media", "version": "1.0"},
            )

            print(f"  ✅ Pipeline '{pipeline.name}' with {pipeline.action_count} actions")

            # Step 4: Execute pipeline
            print("\n🔄 Step 4: Executing pipeline...")
            processed_image = image_asset

            for i, action_spec in enumerate(pipeline.enabled_actions):
                action = registry.get_action(action_spec.name)
                print(f"  🎯 Action {i+1}: {action.label}")

                # Show parameters
                for param_name, param_value in action_spec.parameters.items():
                    print(f"    • {param_name}: {param_value}")

                # Execute
                processed_image = action.execute(processed_image, **action_spec.parameters)

                print(f"    ✅ Result: {processed_image.width}x{processed_image.height}")

            # Step 5: Show results
            print("\n📊 Step 5: Processing Results")
            print(f"  📐 Original: {image_asset.width}x{image_asset.height}")
            print(f"  📐 Final:    {processed_image.width}x{processed_image.height}")
            print(f"  🏷️  Watermark: {processed_image.metadata.get('watermark_text', 'None')}")

            # Show template variables
            print("\n🔧 Template Variables Available:")
            template_vars = image_asset.template_vars
            key_vars = ["filename", "width", "height", "creator", "title", "camera_make", "megapixels"]
            for var in key_vars:
                if var in template_vars:
                    print(f"  • {var}: {template_vars[var]}")

            # Step 6: Test template engine
            print("\n📝 Step 6: Template Engine Examples")
            evaluator = TemplateEvaluator()

            examples = [
                "<filename>_<width>x<height>",
                "upper(<title>)",
                "<creator>_<camera_make>_<megapixels>MP",
                "<width*2>px_wide",
            ]

            for template in examples:
                try:
                    result = evaluator.evaluate(template, template_vars)
                    print(f"  • '{template}' → '{result}'")
                except Exception as e:
                    print(f"  • '{template}' → Error: {e}")

            # Step 7: Pipeline serialization
            print("\n💾 Step 7: Pipeline Persistence")
            pipeline_file = Path(tmpdir) / "demo_pipeline.json"
            save_pipeline(pipeline, pipeline_file)

            loaded_pipeline = load_pipeline(pipeline_file)
            print(f"  ✅ Saved and loaded pipeline: {loaded_pipeline.name}")
            print(f"  📄 Actions: {[action.name for action in loaded_pipeline.actions]}")

            # Step 8: Processing history
            print("\n📜 Step 8: Processing History")
            history = processed_image.processing_metadata.get("processing_history", [])
            for i, step in enumerate(history, 1):
                print(f"  {i}. {step}")

            print("\n🎉 PhotoFlow demonstration completed successfully!")
            print(f"🚀 Processed {image_asset.megapixels:.1f}MP → {processed_image.megapixels:.1f}MP")

    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install Pillow piexif")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    demonstrate_photoflow()
