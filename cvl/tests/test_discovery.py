"""Tests for discovery module."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import sys


from cvl.core.discovery import find_repo_root, load_example_yaml, find_all_examples


class TestFindRepoRoot(unittest.TestCase):
    """Test find_repo_root function."""

    def test_finds_git_root(self):
        """Should find repo root from current directory."""
        root = find_repo_root()
        self.assertTrue((root / ".git").exists())
        self.assertTrue((root / "examples").exists())

    def test_returns_configured_root_when_outside_repo(self):
        """When called from outside the repo, should fall back to cached root."""
        root = find_repo_root()
        with TemporaryDirectory() as tmpdir:
            resolved = find_repo_root(Path(tmpdir))
            self.assertEqual(resolved, root)


class TestLoadExampleYaml(unittest.TestCase):
    """Test load_example_yaml function."""

    def test_loads_valid_yaml(self):
        """Should load valid example.yaml file from repo."""
        # Use an actual example from the repo for testing
        root = find_repo_root()
        examples_dir = root / "examples"

        # Find first example directory with example.yaml
        for yaml_file in examples_dir.rglob("example.yaml"):
            result = load_example_yaml(yaml_file.parent)
            self.assertIsNotNone(result)
            self.assertIn("name", result)
            self.assertIn("_path", result)
            break  # Test with just one example

    def test_returns_none_when_missing(self):
        """Should return None when example.yaml doesn't exist."""
        with TemporaryDirectory() as tmpdir:
            result = load_example_yaml(Path(tmpdir))
            self.assertIsNone(result)


class TestFindAllExamples(unittest.TestCase):
    """Test find_all_examples function."""

    def test_finds_examples_in_repo(self):
        """Should find examples in the actual repository."""
        examples = find_all_examples()
        self.assertGreater(len(examples), 0)
        # Verify structure of first example
        if examples:
            example = examples[0]
            self.assertIn("name", example)
            self.assertIn("_path", example)


if __name__ == "__main__":
    unittest.main()
