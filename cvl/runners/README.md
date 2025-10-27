# CVL Remote Runners

Execute CVL examples on remote machines with automatic setup and cleanup.

## Features

- **SSH Runner**: Run on any remote host via SSH
- **Lambda Labs Runner**: Automated instance lifecycle management
- **Automatic cleanup**: Guaranteed instance termination on exit/error/Ctrl+C
- **Timeout support**: Kill runaway jobs with remote shutdown
- **Cost estimation**: See pricing before launching
- **Progress streaming**: Real-time output from remote jobs

## Installation

```bash
# Install CVL with remote execution support
pip install -e .[remote]

# Or install dependencies manually
pip install paramiko requests
```

## Usage

### SSH Runner (Any Remote Host)

Use this for existing instances (Lambda, AWS, your own server, etc.)

```python
from cvl.runners import SSHRunner

runner = SSHRunner()

runner.run_remote(
    host="ubuntu@123.45.67.89",
    example="nanogpt",
    preset="train",
    args=["--max_iters=1000"],
    timeout_minutes=120,  # Optional: kill after 2 hours
    setup_cvl=True  # Auto-install CVL if needed
)
```

**From command line:**
```bash
# Coming soon: cvl run --ssh command
python -m cvl.runners.example_usage ssh
```

### Lambda Labs Runner (Automated)

Automatically creates instance, runs job, and terminates.

```python
from cvl.runners import LambdaLabsRunner

# Set API key
export LAMBDA_API_KEY=your_key_here

runner = LambdaLabsRunner()

runner.run(
    example="nanogpt",
    preset="train",
    args=["--max_iters=1000", "--batch_size=16"],
    gpu_type="gpu_1x_a100_sxm4",
    timeout_minutes=120
)
```

**From command line:**
```bash
# Coming soon: cvl run --remote lambda command
python -m cvl.runners.example_usage lambda
```

## Configuration

### SSH Authentication

The SSH runner uses your default SSH key (`~/.ssh/id_rsa`). To use a different key:

```python
runner = SSHRunner(ssh_key_path="/path/to/key")
```

### Lambda Labs API Key

Set your API key via environment variable:

```bash
export LAMBDA_API_KEY=your_api_key_here
```

Or pass it directly:

```python
runner = LambdaLabsRunner(api_key="your_key")
```

## Timeout Behavior

Both runners support timeouts:

```python
runner.run_remote(
    ...,
    timeout_minutes=120,  # Kill job after 2 hours
    timeout_action="sudo shutdown -h now"  # What to do on timeout
)
```

**How it works:**
- Remote command runs with `timeout 120m cvl run ...`
- If timeout occurs (exit code 124), runs `timeout_action`
- For Lambda Labs, instance shuts down → billing stops
- Local cleanup always runs regardless

**Exit codes:**
- `0`: Success
- `1-123`: Command error (local cleanup runs)
- `124`: Timeout (remote action runs, then local cleanup)
- `130`: Ctrl+C (local cleanup runs)

## Cost Safety

Users are responsible for monitoring costs. The runner will terminate instances on completion, error, or timeout, but check your Lambda Labs dashboard for any orphaned instances.

## Cleanup Guarantees

**Lambda Labs Runner guarantees cleanup via:**
1. ✅ Try/finally block (normal case)
2. ✅ atexit handler (Python exit)
3. ✅ Signal handlers (Ctrl+C, SIGTERM)
4. ✅ Remote timeout action (runaway jobs)

**Only fails if:**
- ❌ Kill -9 (can't catch)
- ❌ Power outage

For those cases, manually terminate via Lambda dashboard or API.

## Examples

### Train GPT-2 on A100
```python
runner = LambdaLabsRunner()
runner.run(
    example="nanogpt",
    preset="train",
    args=["--max_iters=5000", "--batch_size=32"],
    gpu_type="gpu_1x_a100_sxm4",
    timeout_minutes=180  # 3 hour safety net
)
```

### Run Inference on Existing Instance
```python
runner = SSHRunner()
runner.run_remote(
    host="ubuntu@my-lambda.lambdalabs.com",
    example="mixtral8x7b",
    preset="predict",
    args=[],
    setup_cvl=False  # Already installed
)
```

### Quick Test Run
```python
runner = LambdaLabsRunner()
runner.run("nanogpt", "train", ["--max_iters=100"], timeout_minutes=10)
```

## Architecture

```
┌─────────────────┐
│ LambdaLabsRunner│  (Lifecycle: create → wait → run → terminate)
└────────┬────────┘
         │ uses
         ▼
┌─────────────────┐
│   SSHRunner     │  (Execution: connect → setup → command → stream)
└─────────────────┘
```

**Lambda Labs Runner:**
- Instance creation via API
- Wait for SSH ready
- Delegates execution to SSH runner
- Handles cleanup

**SSH Runner:**
- Works with any SSH host
- Auto-installs CVL if needed
- Proper argument escaping (prevents shell injection)
- Timeout with remote actions
- Real-time output streaming

## Troubleshooting

**"paramiko not installed"**
```bash
pip install paramiko
```

**"LAMBDA_API_KEY not set"**
```bash
export LAMBDA_API_KEY=your_key
```

**"Authentication failed"**
- Check your SSH key: `ssh-add -l`
- Add key to remote: `ssh-copy-id ubuntu@host`

**"Instance not ready after 300s"**
- Lambda may be slow or out of capacity
- Try different region or GPU type

**"Instance not terminated"**
- Check Lambda dashboard
- Manually terminate if needed
- Check for orphaned instances regularly

## Future Enhancements

- [ ] CLI integration: `cvl run --ssh host example preset`
- [ ] CLI integration: `cvl run --remote lambda example preset`
- [ ] State file for orphaned instance cleanup
- [ ] Support for other cloud providers (AWS, GCP, RunPod)
- [ ] Heartbeat pattern for bulletproof cleanup
- [ ] Artifact download (save checkpoints locally)
- [ ] Multi-instance support (parallel runs)

## Development

Run tests:
```bash
pytest cvl/tests/test_runner.py
```

See `example_usage.py` for more examples.
