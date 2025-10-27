# Publishing CVL to PyPI

This document describes how to publish the `cvl` CLI package to PyPI.

## Package Overview

- **Package name**: `cvl` (on PyPI)
- **Distribution**: Wheel-only (no examples included)
- **Size**: ~19KB
- **Dependencies**: PyYAML only

## Pre-requisites

1. Install build tools:
   ```bash
   pip install build twine
   ```

2. Set up PyPI credentials:
   - Create account on https://pypi.org
   - Generate API token at https://pypi.org/manage/account/token/
   - Store in `~/.pypirc`:
     ```ini
     [pypi]
     username = __token__
     password = pypi-...your-token...
     ```

## Release Process

### 1. Update Version

Before building, update the version in two places:

**cvl/_version.py:**
```python
__version__ = "0.2.0"  # Update version
__git_hash__ = "abc123f"  # Update to current commit hash
```

**pyproject.toml:**
```toml
[project]
version = "0.2.0"  # Match _version.py
```

Get the current git hash:
```bash
git rev-parse --short HEAD
```

### 2. Build the Wheel

```bash
# Clean previous builds
rm -rf dist/ build/

# Build wheel
python -m build --wheel

# Verify contents and size
ls -lh dist/*.whl
unzip -l dist/*.whl | head -20
```

Expected size: ~19KB
Expected contents: cvl/, cvl/commands/, cvl/core/ (no examples/)

### 3. Test Locally

Test the wheel in a clean environment before publishing:

```bash
# Create test environment
python -m venv /tmp/test_cvl
source /tmp/test_cvl/bin/activate

# Install from wheel
pip install dist/cvl-*.whl

# Test basic functionality
cvl --version

# Set CVLIZATION_ROOT and test commands
export CVLIZATION_ROOT=/path/to/CVlization
cvl list --stability stable
cvl info perception/ocr_and_layout/surya

# Clean up
deactivate
rm -rf /tmp/test_cvl
```

### 4. Publish to Test PyPI (Optional)

Test the upload process on Test PyPI first:

```bash
# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ cvl
```

### 5. Publish to PyPI

```bash
# Upload to production PyPI
python -m twine upload dist/*

# Verify on PyPI
# Visit https://pypi.org/project/cvl/
```

### 6. Tag the Release

```bash
# Create git tag
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0

# Create GitHub release (optional)
# Visit https://github.com/kungfuai/CVlization/releases/new
```

## User Installation

After publishing, users can install with:

```bash
# Install CLI globally with pipx (recommended)
pipx install cvl

# Or with pip
pip install cvl
```

Then clone the examples repository:

```bash
# Full clone
git clone https://github.com/kungfuai/CVlization
cd CVlization

# Or shallow clone (faster, no history)
git clone --depth 1 https://github.com/kungfuai/CVlization
cd CVlization
```

Usage:
```bash
# CLI works from inside the repo
cvl list
cvl info perception/ocr_and_layout/surya
cvl run perception/ocr_and_layout/surya build

# Or set CVLIZATION_ROOT to use from anywhere
export CVLIZATION_ROOT=/path/to/CVlization
cvl list
```

## Versioning Strategy

- **Semantic versioning**: MAJOR.MINOR.PATCH
- **Git hash baked in**: Every build includes the commit hash for traceability
- **Version in two places**: pyproject.toml and _version.py (must match)

## Troubleshooting

### Build fails with "Invalid license"

This is a deprecation warning, not an error. The build still succeeds. To fix, update pyproject.toml:
```toml
[project]
license = "MIT"  # Instead of {text = "MIT"}
```

### Wheel includes test files

Tests are currently included (~21KB extra). This is not critical but can be optimized by moving `cvl/tests/` to `tests/` at the repo root.

### Upload fails with authentication error

1. Verify `~/.pypirc` contains valid API token
2. Check token hasn't expired
3. Ensure username is `__token__` (exactly)

## Automation (Future)

Consider GitHub Actions workflow for automated releases:

```yaml
name: Publish to PyPI
on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install build twine
      - run: python -m build --wheel
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

## Notes

- **Separation of concerns**: CLI package (PyPI) is separate from examples repository (GitHub)
- **Lightweight distribution**: ~19KB wheel downloads in <1 second
- **No framework coupling**: Users clone full repo to get examples, CLI just orchestrates
- **Version pinning**: Git hash allows tracing CLI version back to exact commit
