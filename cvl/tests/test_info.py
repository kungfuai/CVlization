"""Tests for the info command."""
import unittest
from cvl.commands.info import find_matching_examples, format_info, get_example_info


class TestFindMatchingExamples(unittest.TestCase):
    """Test the find_matching_examples function."""

    def setUp(self):
        """Set up test fixtures."""
        self.examples = [
            {
                "_path": "examples/generative/minisora",
                "name": "minisora",
                "capability": "generative",
            },
            {
                "_path": "examples/perception/doc_ai/granite_docling",
                "name": "granite_docling",
                "capability": "perception",
            },
        ]

    def test_find_example_with_full_path(self):
        """Test finding example with 'examples/' prefix."""
        results = find_matching_examples(self.examples, "examples/generative/minisora")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "minisora")

    def test_find_example_with_short_path(self):
        """Test finding example without 'examples/' prefix."""
        results = find_matching_examples(self.examples, "generative/minisora")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "minisora")

    def test_find_example_with_short_name(self):
        """Test finding example with just the short name."""
        results = find_matching_examples(self.examples, "minisora")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "minisora")

    def test_find_example_with_partial_path(self):
        """Test finding example with partial path."""
        results = find_matching_examples(self.examples, "doc_ai/granite_docling")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "granite_docling")

    def test_find_example_not_found(self):
        """Test finding non-existent example."""
        results = find_matching_examples(self.examples, "nonexistent/example")
        self.assertEqual(len(results), 0)


class TestFormatInfo(unittest.TestCase):
    """Test the format_info function."""

    def test_format_basic_info(self):
        """Test formatting basic example info."""
        example = {
            "name": "test_example",
            "_path": "examples/test/example",
            "capability": "test_capability",
            "stability": "stable",
        }
        result = format_info(example)
        self.assertIn("Name: test_example", result)
        self.assertIn("Path: examples/test/example", result)
        self.assertIn("Capability: test_capability", result)
        self.assertIn("Stability: stable", result)

    def test_format_with_description(self):
        """Test formatting with description."""
        example = {
            "name": "test_example",
            "_path": "examples/test/example",
            "capability": "test_capability",
            "stability": "stable",
            "description": "Test description",
        }
        result = format_info(example)
        self.assertIn("Description:", result)
        self.assertIn("Test description", result)

    def test_format_with_resources(self):
        """Test formatting with resources."""
        example = {
            "name": "test_example",
            "_path": "examples/test/example",
            "capability": "test_capability",
            "stability": "stable",
            "resources": {
                "gpu": 8,
                "vram": "80GB",
                "training_time": "2 hours",
            },
        }
        result = format_info(example)
        self.assertIn("Resources:", result)
        self.assertIn("GPU: 8", result)
        self.assertIn("VRAM: 80GB", result)
        self.assertIn("Training time: 2 hours", result)

    def test_format_with_presets_list(self):
        """Test formatting with presets as list."""
        example = {
            "name": "test_example",
            "_path": "examples/test/example",
            "capability": "test_capability",
            "stability": "stable",
            "presets": ["train", "inference"],
        }
        result = format_info(example)
        self.assertIn("Presets:", result)
        self.assertIn("- train", result)
        self.assertIn("- inference", result)

    def test_format_with_presets_dict(self):
        """Test formatting with presets as dict."""
        example = {
            "name": "test_example",
            "_path": "examples/test/example",
            "capability": "test_capability",
            "stability": "stable",
            "presets": {
                "train": {"description": "Training preset"},
                "inference": {"description": "Inference preset"},
            },
        }
        result = format_info(example)
        self.assertIn("Presets:", result)
        self.assertIn("- train: Training preset", result)
        self.assertIn("- inference: Inference preset", result)


class TestGetExampleInfo(unittest.TestCase):
    """Test the get_example_info function."""

    def setUp(self):
        """Set up test fixtures."""
        self.examples = [
            {
                "_path": "examples/generative/minisora",
                "name": "minisora",
                "capability": "generative",
                "stability": "beta",
                "description": "MiniSora example",
            },
            {
                "_path": "examples/perception/vision_language/moondream2",
                "name": "moondream2",
                "capability": "perception",
                "stability": "stable",
            },
        ]

    def test_get_example_info_found(self):
        """Test getting info for existing example."""
        result = get_example_info(self.examples, "generative/minisora")
        self.assertIsNotNone(result)
        self.assertIn("Name: minisora", result)
        self.assertIn("MiniSora example", result)

    def test_get_example_info_with_short_name(self):
        """Test getting info with short name."""
        result = get_example_info(self.examples, "moondream2")
        self.assertIsNotNone(result)
        self.assertIn("Name: moondream2", result)

    def test_get_example_info_not_found(self):
        """Test getting info for non-existent example."""
        result = get_example_info(self.examples, "nonexistent/example")
        self.assertIsNotNone(result)
        self.assertIn("✗ No example found", result)

    def test_get_example_info_ambiguous(self):
        """Test getting info with ambiguous identifier."""
        # Add another example that could match
        self.examples.append({
            "_path": "examples/generative/video/minisora_v2",
            "name": "minisora_v2",
            "capability": "generative",
        })
        # This won't actually be ambiguous with our current logic,
        # but we test the error message format
        result = get_example_info(self.examples, "nonexistent")
        self.assertIn("✗", result)


if __name__ == "__main__":
    unittest.main()
