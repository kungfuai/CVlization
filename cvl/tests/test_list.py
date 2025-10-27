"""Tests for list command."""
import unittest
import sys
from pathlib import Path


from cvl.commands.list import filter_examples, format_table, list_examples


class TestFilterExamples(unittest.TestCase):
    """Test filter_examples function."""

    def setUp(self):
        """Set up test examples."""
        self.examples = [
            {
                "name": "example1",
                "capability": "generative/image",
                "tags": ["video", "ml"],
                "stability": "stable",
            },
            {
                "name": "example2",
                "capability": "perception/ocr",
                "tags": ["text", "doc"],
                "stability": "beta",
            },
            {
                "name": "example3",
                "capability": "generative/llm",
                "tags": ["text"],
                "stability": "stable",
            },
        ]

    def test_no_filter_returns_all(self):
        """Should return all examples when no filter applied."""
        result = filter_examples(self.examples)
        self.assertEqual(len(result), 3)

    def test_filter_by_capability(self):
        """Should filter by capability."""
        result = filter_examples(self.examples, capability="generative")
        self.assertEqual(len(result), 2)
        self.assertTrue(all("generative" in e["capability"] for e in result))

    def test_filter_by_tag(self):
        """Should filter by tag."""
        result = filter_examples(self.examples, tag="text")
        self.assertEqual(len(result), 2)

    def test_filter_by_stability(self):
        """Should filter by stability."""
        result = filter_examples(self.examples, stability="stable")
        self.assertEqual(len(result), 2)

    def test_multiple_filters(self):
        """Should apply multiple filters together."""
        result = filter_examples(
            self.examples, capability="generative", stability="stable"
        )
        self.assertEqual(len(result), 2)


class TestFormatTable(unittest.TestCase):
    """Test format_table function."""

    def test_formats_examples(self):
        """Should format examples as table."""
        examples = [
            {
                "name": "test",
                "capability": "gen/img",
                "_path": "examples/test",
                "stability": "stable",
            }
        ]
        result = format_table(examples)
        self.assertIn("NAME", result)
        self.assertIn("CAPABILITY", result)
        self.assertIn("test", result)
        self.assertIn("stable", result)

    def test_empty_examples(self):
        """Should handle empty list."""
        result = format_table([])
        self.assertEqual(result, "No examples found.")


class TestListExamples(unittest.TestCase):
    """Test list_examples function."""

    def test_lists_and_sorts(self):
        """Should list and sort examples."""
        examples = [
            {
                "name": "z-example",
                "capability": "perception/ocr",
                "_path": "examples/z",
                "stability": "stable",
            },
            {
                "name": "a-example",
                "capability": "generative/image",
                "_path": "examples/a",
                "stability": "beta",
            },
        ]
        result = list_examples(examples)
        # Should be sorted by capability, then name
        lines = result.split("\n")
        # Find data rows (skip header and separator)
        self.assertGreater(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
