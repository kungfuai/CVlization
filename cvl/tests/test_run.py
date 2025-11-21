"""Tests for the run command."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from cvl.commands.run import (
    get_preset_info,
    find_script,
    get_example_path,
    run_script,
    run_example,
)


class TestGetPresetInfo(unittest.TestCase):
    """Test the get_preset_info function."""

    def test_dict_format_with_full_info(self):
        """Test getting preset info from dict format with all fields."""
        example = {
            "presets": {
                "train": {
                    "script": "train.sh",
                    "description": "Train the model",
                }
            }
        }
        result = get_preset_info(example, "train")
        self.assertIsNotNone(result)
        self.assertEqual(result["script"], "train.sh")
        self.assertEqual(result["description"], "Train the model")

    def test_dict_format_with_script_only(self):
        """Test getting preset info from dict format with script only."""
        example = {"presets": {"train": {"script": "custom_train.sh"}}}
        result = get_preset_info(example, "train")
        self.assertIsNotNone(result)
        self.assertEqual(result["script"], "custom_train.sh")
        self.assertEqual(result["description"], "")

    def test_dict_format_with_string_value(self):
        """Test getting preset info from dict format with string value."""
        example = {"presets": {"train": "my_script.sh"}}
        result = get_preset_info(example, "train")
        self.assertIsNotNone(result)
        self.assertEqual(result["script"], "my_script.sh")
        self.assertEqual(result["description"], "")

    def test_list_format_with_convention(self):
        """Test getting preset info from list format (uses convention)."""
        example = {"presets": ["train", "predict"]}
        result = get_preset_info(example, "train")
        self.assertIsNotNone(result)
        self.assertEqual(result["script"], "train.sh")
        self.assertEqual(result["description"], "")

    def test_preset_not_found_in_dict(self):
        """Test getting preset info when not found in dict format."""
        example = {"presets": {"train": {"script": "train.sh"}}}
        result = get_preset_info(example, "nonexistent")
        self.assertIsNone(result)

    def test_preset_not_found_in_list(self):
        """Test getting preset info when not found in list format."""
        example = {"presets": ["train", "predict"]}
        result = get_preset_info(example, "nonexistent")
        self.assertIsNone(result)

    def test_no_presets_field(self):
        """Test getting preset info when no presets field exists."""
        example = {"name": "test"}
        result = get_preset_info(example, "train")
        self.assertIsNone(result)


class TestFindScript(unittest.TestCase):
    """Test the find_script function."""

    def test_find_existing_script(self):
        """Test finding an existing script file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a script file
            script_path = os.path.join(tmpdir, "train.sh")
            with open(script_path, "w") as f:
                f.write("#!/bin/bash\necho 'test'")

            result = find_script(tmpdir, "train.sh")
            self.assertEqual(result, script_path)

    def test_script_not_found(self):
        """Test when script file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_script(tmpdir, "nonexistent.sh")
            self.assertIsNone(result)


class TestGetExamplePath(unittest.TestCase):
    """Test the get_example_path function."""

    def setUp(self):
        """Set up test fixtures."""
        self.examples = [
            {
                "_path": "examples/generative/minisora",
                "name": "minisora",
            },
            {
                "_path": "examples/perception/doc_ai",
                "name": "granite_docling",
            },
        ]

    @patch("cvl.core.discovery.find_repo_root")
    def test_get_path_with_full_prefix(self, mock_find_repo_root):
        """Test getting path with 'examples/' prefix."""
        mock_find_repo_root.return_value = Path("/repo")
        result = get_example_path(self.examples, "examples/generative/minisora")
        self.assertEqual(result, "/repo/examples/generative/minisora")

    @patch("cvl.core.discovery.find_repo_root")
    def test_get_path_without_prefix(self, mock_find_repo_root):
        """Test getting path without 'examples/' prefix."""
        mock_find_repo_root.return_value = Path("/repo")
        result = get_example_path(self.examples, "generative/minisora")
        self.assertEqual(result, "/repo/examples/generative/minisora")

    def test_get_path_not_found(self):
        """Test getting path for non-existent example."""
        result = get_example_path(self.examples, "nonexistent/example")
        self.assertIsNone(result)


class TestRunScript(unittest.TestCase):
    """Test the run_script function."""

    def test_script_not_found(self):
        """Test running a non-existent script."""
        exit_code, error_msg = run_script("/nonexistent/script.sh", [])
        self.assertEqual(exit_code, 1)
        self.assertIn("Script not found", error_msg)

    def test_script_not_executable(self):
        """Test running a non-executable script."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test.sh")
            with open(script_path, "w") as f:
                f.write("#!/bin/bash\necho 'test'")
            # Don't make it executable

            exit_code, error_msg = run_script(script_path, [])
            self.assertEqual(exit_code, 1)
            self.assertIn("not executable", error_msg)

    @patch("subprocess.run")
    def test_successful_script_execution(self, mock_run):
        """Test successful script execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test.sh")
            with open(script_path, "w") as f:
                f.write("#!/bin/bash\necho 'test'")
            os.chmod(script_path, 0o755)

            # Mock successful execution
            mock_run.return_value = Mock(returncode=0)

            exit_code, error_msg = run_script(script_path, [], no_live=True)
            self.assertEqual(exit_code, 0)
            self.assertEqual(error_msg, "")
            mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_script_execution_with_args(self, mock_run):
        """Test script execution with additional arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test.sh")
            with open(script_path, "w") as f:
                f.write("#!/bin/bash\necho 'test'")
            os.chmod(script_path, 0o755)

            # Mock successful execution
            mock_run.return_value = Mock(returncode=0)

            exit_code, error_msg = run_script(script_path, ["--arg1", "value1"], no_live=True)
            self.assertEqual(exit_code, 0)
            self.assertEqual(error_msg, "")

            # Verify args were passed
            call_args = mock_run.call_args
            self.assertIn("--arg1", call_args[0][0])
            self.assertIn("value1", call_args[0][0])

    @patch("subprocess.run")
    def test_run_script_creates_cache_dirs(self, mock_run):
        """Ensure HF/Torch cache dirs are created before docker mounts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test.sh")
            with open(script_path, "w") as f:
                f.write("#!/bin/bash\nexit 0")
            os.chmod(script_path, 0o755)

            mock_run.return_value = Mock(returncode=0)

            with patch.dict(os.environ, {"HOME": tmpdir}, clear=False):
                exit_code, _ = run_script(script_path, [], no_live=True)

            self.assertEqual(exit_code, 0)
            env_passed = mock_run.call_args.kwargs["env"]

            for var in ["HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME"]:
                path = Path(env_passed[var])
                self.assertTrue(path.exists(), f"{var} directory should exist")

class TestRunExample(unittest.TestCase):
    """Test the run_example function."""

    def setUp(self):
        """Set up test fixtures."""
        self.examples = [
            {
                "_path": "examples/generative/test_example",
                "name": "test_example",
                "presets": {
                    "train": {"script": "train.sh", "description": "Train model"}
                },
            },
            {
                "_path": "examples/generative/old_format",
                "name": "old_format",
                "presets": ["train", "predict"],
            },
        ]

    @patch("cvl.commands.run.check_docker_running", return_value=(True, ""))
    def test_example_not_found(self, _mock_docker):
        """Test running non-existent example."""
        exit_code, error_msg = run_example(
            self.examples, "nonexistent/example", "train"
        )
        self.assertEqual(exit_code, 1)
        self.assertIn("not found", error_msg)

    @patch("cvl.commands.run.check_docker_running", return_value=(True, ""))
    def test_preset_not_found(self, _mock_docker):
        """Test running with non-existent preset."""
        exit_code, error_msg = run_example(
            self.examples, "generative/test_example", "nonexistent"
        )
        self.assertEqual(exit_code, 1)
        self.assertIn("Preset 'nonexistent' not found", error_msg)
        self.assertIn("Available: train", error_msg)

    @patch("cvl.commands.run.check_docker_running", return_value=(True, ""))
    @patch("cvl.commands.run.find_script")
    def test_script_not_found(self, mock_find_script, _mock_docker):
        """Test when script file doesn't exist."""
        mock_find_script.return_value = None

        exit_code, error_msg = run_example(
            self.examples, "generative/test_example", "train"
        )
        self.assertEqual(exit_code, 1)
        self.assertIn("Script not found", error_msg)

    @patch("cvl.commands.run.check_docker_image_exists", return_value=True)
    @patch("cvl.commands.run.check_docker_running", return_value=(True, ""))
    @patch("cvl.commands.run.run_script")
    @patch("cvl.commands.run.find_script")
    def test_successful_run(self, mock_find_script, mock_run_script, _mock_docker, _mock_image):
        """Test successful example run."""
        mock_find_script.return_value = "/path/to/train.sh"
        mock_run_script.return_value = (0, "")

        exit_code, error_msg = run_example(
            self.examples, "generative/test_example", "train"
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(error_msg, "")

        mock_run_script.assert_called_once()
        args, kwargs = mock_run_script.call_args
        self.assertEqual(args[0], "/path/to/train.sh")
        self.assertEqual(args[1], [])
        self.assertFalse(kwargs.get("no_live"))
        self.assertEqual(kwargs.get("job_name"), "test_example train")
        self.assertEqual(kwargs.get("image_name"), "test_example")
        self.assertIsNone(kwargs.get("work_dir"))

    @patch("cvl.commands.run.check_docker_image_exists", return_value=True)
    @patch("cvl.commands.run.check_docker_running", return_value=(True, ""))
    @patch("cvl.commands.run.run_script")
    @patch("cvl.commands.run.find_script")
    def test_run_with_extra_args(self, mock_find_script, mock_run_script, _mock_docker, _mock_image):
        """Test running example with extra arguments."""
        mock_find_script.return_value = "/path/to/train.sh"
        mock_run_script.return_value = (0, "")

        exit_code, error_msg = run_example(
            self.examples,
            "generative/test_example",
            "train",
            ["--epochs", "10"],
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(error_msg, "")

        mock_run_script.assert_called_once()
        args, kwargs = mock_run_script.call_args
        self.assertEqual(args[0], "/path/to/train.sh")
        self.assertEqual(args[1], ["--epochs", "10"])
        self.assertFalse(kwargs.get("no_live"))

    @patch("cvl.commands.run.check_docker_image_exists", return_value=True)
    @patch("cvl.commands.run.check_docker_running", return_value=(True, ""))
    @patch("cvl.commands.run.run_script")
    @patch("cvl.commands.run.find_script")
    def test_path_args_env_passed_to_run_script(self, mock_find_script, mock_run_script, _mock_docker, _mock_image):
        """Test that path_args_env is properly passed from run_example to run_script."""
        mock_find_script.return_value = "/path/to/train.sh"
        mock_run_script.return_value = (0, "")

        exit_code, error_msg = run_example(
            self.examples,
            "generative/test_example",
            "train",
            [],
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(error_msg, "")

        # Verify path_args_env is passed to run_script
        mock_run_script.assert_called_once()
        args, kwargs = mock_run_script.call_args
        self.assertIn("path_args_env", kwargs)
        # Should be passed even if None/empty
        self.assertIsInstance(kwargs.get("path_args_env"), (dict, type(None)))


if __name__ == "__main__":
    unittest.main()
