"""SageMaker Training Job runner for CVL examples."""

import atexit
import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, List


class SageMakerRunner:
    """
    Run CVL examples on AWS SageMaker Training with automatic container management.

    Features:
    - Auto-builds Docker image from example's Dockerfile
    - Pushes to ECR (creates repository if needed)
    - Spot instance support for cost savings
    - Automatic artifact download from S3
    - CloudWatch log streaming
    - Cleanup on exit/interrupt

    Requirements:
    - AWS credentials configured (boto3)
    - Docker installed locally (for building images)
    - ECR permissions (ecr:CreateRepository, ecr:GetAuthorizationToken, etc.)
    - SageMaker permissions (sagemaker:CreateTrainingJob, etc.)
    - IAM role for SageMaker execution

    Example:
        runner = SageMakerRunner(role_arn="arn:aws:iam::123456789:role/SageMakerRole")
        runner.run(
            example="nanogpt",
            preset="train",
            args=["--max_iters=1000"],
            instance_type="ml.g5.xlarge",
            output_path="s3://my-bucket/outputs/",
        )
    """

    def __init__(
        self,
        role_arn: str,
        region: Optional[str] = None,
        ecr_repo_prefix: str = "cvl",
    ):
        """
        Initialize SageMaker runner.

        Args:
            role_arn: IAM role ARN for SageMaker execution.
                      Must have permissions to access S3, ECR, CloudWatch.
            region: AWS region (defaults to boto3 default or AWS_DEFAULT_REGION)
            ecr_repo_prefix: Prefix for ECR repository names (default: "cvl")
        """
        try:
            import boto3
        except ImportError:
            print("boto3 not installed. Install with: pip install boto3")
            sys.exit(1)

        self.role_arn = role_arn
        self.ecr_repo_prefix = ecr_repo_prefix
        self.region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")

        # Initialize AWS clients
        self.sagemaker = boto3.client("sagemaker", region_name=self.region)
        self.ecr = boto3.client("ecr", region_name=self.region)
        self.logs = boto3.client("logs", region_name=self.region)
        self.sts = boto3.client("sts", region_name=self.region)

        # Get account ID for ECR URI
        self.account_id = self.sts.get_caller_identity()["Account"]

        # Track current job for cleanup
        self.current_job_name = None

        # Setup cleanup handlers
        self._setup_cleanup_handlers()

    def run(
        self,
        example: str,
        preset: str,
        args: List[str],
        instance_type: str,
        output_path: str,
        input_path: Optional[str] = None,
        instance_count: int = 1,
        spot: bool = False,
        max_wait_minutes: Optional[int] = None,
        max_run_minutes: int = 60,
        volume_size_gb: int = 50,
        image_uri: Optional[str] = None,
        entry_command: str = "python train.py",
        download_outputs: bool = True,
        local_output_dir: Optional[str] = None,
    ) -> int:
        """
        Run CVL example on SageMaker.

        Args:
            example: Example name (e.g., "nanogpt")
            preset: Preset name (e.g., "train")
            args: Additional arguments for the training script
            instance_type: SageMaker instance type (e.g., "ml.g5.xlarge")
            output_path: S3 path for outputs (e.g., "s3://bucket/outputs/")

            input_path: Optional S3 path for input data
            instance_count: Number of instances (default: 1)
            spot: Use spot instances for cost savings (default: False)
            max_wait_minutes: Max wait time for spot (required if spot=True)
            max_run_minutes: Max training time in minutes (default: 60)
            volume_size_gb: EBS volume size (default: 50)

            image_uri: Override container image (default: auto-build from Dockerfile)
            entry_command: Training command to run (default: "python train.py")
                          Examples: "python train.py", "torchrun --nproc_per_node=4 train.py"
            download_outputs: Download outputs from S3 after completion (default: True)
            local_output_dir: Local directory for outputs (default: example/outputs/)

        Returns:
            Exit code (0 for success)
        """
        if spot and not max_wait_minutes:
            raise ValueError("max_wait_minutes required when using spot instances")

        try:
            # Find example directory
            example_dir = self._find_example(example)
            print(f"Found example: {example_dir}")

            # Build and push container (or use provided image)
            if image_uri:
                container_uri = image_uri
                print(f"Using provided image: {container_uri}")
            else:
                container_uri = self._build_and_push(example, example_dir)

            # Create training job
            job_name = self._create_job_name(example, preset)
            self.current_job_name = job_name

            print(f"Creating training job: {job_name}")
            self._create_training_job(
                job_name=job_name,
                image_uri=container_uri,
                example=example,
                preset=preset,
                args=args,
                entry_command=entry_command,
                example_dir=example_dir,
                instance_type=instance_type,
                instance_count=instance_count,
                output_path=output_path,
                input_path=input_path,
                spot=spot,
                max_wait_minutes=max_wait_minutes,
                max_run_minutes=max_run_minutes,
                volume_size_gb=volume_size_gb,
            )

            # Monitor job
            exit_code = self._monitor_job(job_name)

            # Download outputs
            if download_outputs and exit_code == 0:
                output_dir = local_output_dir or str(example_dir / "outputs")
                self._download_outputs(output_path, job_name, output_dir)

            return exit_code

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            return 130

        except Exception as e:
            print(f"Error: {e}")
            return 1

        finally:
            self.current_job_name = None

    def _find_example(self, example: str) -> Path:
        """Find example directory by name."""
        # Get CVL root (go up from runners/ to cvl/ to repo root)
        cvl_root = Path(__file__).parent.parent.parent

        # Search in examples/
        examples_dir = cvl_root / "examples"
        if not examples_dir.exists():
            raise FileNotFoundError(f"Examples directory not found: {examples_dir}")

        # Search recursively for example.yaml with matching name
        for yaml_path in examples_dir.rglob("example.yaml"):
            try:
                import yaml
                with open(yaml_path) as f:
                    config = yaml.safe_load(f)
                if config.get("name") == example:
                    return yaml_path.parent
            except Exception:
                continue

        raise FileNotFoundError(f"Example not found: {example}")

    def _build_and_push(self, example: str, example_dir: Path) -> str:
        """
        Build Docker image with SageMaker entry script and push to ECR.

        The build process:
        1. Build original Dockerfile as base image
        2. Create wrapper that extends base with entry script
        3. Push final image to ECR
        """
        repo_name = f"{self.ecr_repo_prefix}-{example}"
        ecr_uri = f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/{repo_name}"

        # Create ECR repository if needed
        self._ensure_ecr_repo(repo_name)

        # Get ECR login token
        print("Authenticating with ECR...")
        auth = self.ecr.get_authorization_token()
        token = auth["authorizationData"][0]["authorizationToken"]
        endpoint = auth["authorizationData"][0]["proxyEndpoint"]

        # Decode token (base64 encoded "AWS:password")
        decoded = base64.b64decode(token).decode()
        password = decoded.split(":")[1]

        # Docker login to ECR
        login_cmd = f"echo {password} | docker login --username AWS --password-stdin {endpoint}"
        result = subprocess.run(login_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ECR login failed: {result.stderr}")

        # Step 1: Build original Dockerfile as base image
        base_tag = f"cvl-{example}-base:latest"
        print(f"Building base image from {example_dir}/Dockerfile...")
        build_cmd = ["docker", "build", "-t", base_tag, str(example_dir)]
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed: {result.stderr}")
        print("Base image built")

        # Step 2: Create wrapper with entry script
        final_tag = f"{ecr_uri}:latest"
        self._build_sagemaker_wrapper(base_tag, final_tag, example_dir)

        # Step 3: Push to ECR
        print(f"Pushing to {ecr_uri}...")
        push_cmd = ["docker", "push", final_tag]
        result = subprocess.run(push_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Docker push failed: {result.stderr}")
        print("Push complete")

        return final_tag

    def _build_sagemaker_wrapper(self, base_tag: str, final_tag: str, example_dir: Path):
        """Build wrapper image that adds SageMaker entry script to base image."""
        # Get the entry script path
        entry_script = Path(__file__).parent / "sagemaker_entry.py"
        if not entry_script.exists():
            raise FileNotFoundError(f"Entry script not found: {entry_script}")

        # Get cvlization package path (relative to this file)
        cvl_root = Path(__file__).parent.parent.parent
        cvlization_pkg = cvl_root / "cvlization"

        # Create temp directory for wrapper build
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Copy entry script
            shutil.copy(entry_script, tmpdir / "sagemaker_entry.py")

            # Copy example directory contents (for example.yaml, configs, etc.)
            example_copy = tmpdir / "example"
            shutil.copytree(example_dir, example_copy, dirs_exist_ok=True)

            # Copy cvlization package if it exists (needed for examples that import it)
            # Filter out unnecessary files to keep image small
            cvlization_copy = tmpdir / "cvlization"
            if cvlization_pkg.exists():
                def ignore_patterns(directory, files):
                    """Ignore pycache, tests, and other non-essential files."""
                    ignored = []
                    for f in files:
                        # Ignore pycache and compiled files
                        if f == "__pycache__" or f.endswith(".pyc") or f.endswith(".pyo"):
                            ignored.append(f)
                        # Ignore test files
                        elif f.startswith("test_") or f == "tests":
                            ignored.append(f)
                        # Ignore large data files
                        elif f.endswith((".pkl", ".bin", ".pt", ".pth", ".ckpt")):
                            ignored.append(f)
                    return ignored
                shutil.copytree(cvlization_pkg, cvlization_copy, ignore=ignore_patterns, dirs_exist_ok=True)

            # Create wrapper Dockerfile
            wrapper_dockerfile = tmpdir / "Dockerfile"
            has_cvlization = cvlization_pkg.exists()
            wrapper_dockerfile.write_text(f"""\
FROM {base_tag}

# Copy SageMaker entry script
COPY sagemaker_entry.py /opt/ml/code/sagemaker_entry.py

# Copy example files to workspace (in case not already there)
COPY example/ /workspace/

{"# Copy cvlization package for imports" if has_cvlization else ""}
{"COPY cvlization/ /workspace/cvlization/" if has_cvlization else ""}

# Add cvlization to Python path
ENV PYTHONPATH="/workspace:$PYTHONPATH"

# Set working directory
WORKDIR /workspace

# Set unbuffered output for real-time logs
ENV PYTHONUNBUFFERED=1

# Entry point for SageMaker
ENTRYPOINT ["python", "/opt/ml/code/sagemaker_entry.py"]
""")

            # Build wrapper
            print("Building SageMaker wrapper image...")
            build_cmd = ["docker", "build", "-t", final_tag, str(tmpdir)]
            result = subprocess.run(build_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Wrapper build failed: {result.stderr}")
            print("Wrapper image built")

    def _ensure_ecr_repo(self, repo_name: str):
        """Create ECR repository if it doesn't exist."""
        try:
            self.ecr.describe_repositories(repositoryNames=[repo_name])
            print(f"ECR repository exists: {repo_name}")
        except self.ecr.exceptions.RepositoryNotFoundException:
            print(f"Creating ECR repository: {repo_name}")
            self.ecr.create_repository(
                repositoryName=repo_name,
                imageScanningConfiguration={"scanOnPush": False},
            )

    def _create_job_name(self, example: str, preset: str) -> str:
        """Generate unique job name."""
        timestamp = int(time.time())
        return f"cvl-{example}-{preset}-{timestamp}"

    def _create_training_job(
        self,
        job_name: str,
        image_uri: str,
        example: str,
        preset: str,
        args: List[str],
        entry_command: str,
        example_dir: Path,
        instance_type: str,
        instance_count: int,
        output_path: str,
        input_path: Optional[str],
        spot: bool,
        max_wait_minutes: Optional[int],
        max_run_minutes: int,
        volume_size_gb: int,
    ):
        """Create SageMaker training job."""
        # Build hyperparameters (passed to entry script via /opt/ml/input/config/)
        # SageMaker requires all values to be strings
        hyperparameters = {
            "example": example,
            "preset": preset,
            "command": entry_command,
            "args": json.dumps(args),
        }

        # Resource config
        resource_config = {
            "InstanceType": instance_type,
            "InstanceCount": instance_count,
            "VolumeSizeInGB": volume_size_gb,
        }

        # Stopping condition
        stopping_condition = {
            "MaxRuntimeInSeconds": max_run_minutes * 60,
        }
        if spot and max_wait_minutes:
            stopping_condition["MaxWaitTimeInSeconds"] = max_wait_minutes * 60

        # Input data config
        input_data_config = []
        if input_path:
            input_data_config.append({
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": input_path,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            })

        # Output data config
        output_data_config = {
            "S3OutputPath": output_path,
        }

        # Algorithm specification
        # Entry point is baked into the container image (see _build_sagemaker_wrapper)
        algorithm_spec = {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        }

        # Create job request
        create_params = {
            "TrainingJobName": job_name,
            "AlgorithmSpecification": algorithm_spec,
            "RoleArn": self.role_arn,
            "HyperParameters": hyperparameters,
            "ResourceConfig": resource_config,
            "StoppingCondition": stopping_condition,
            "OutputDataConfig": output_data_config,
            "EnableManagedSpotTraining": spot,
        }

        if input_data_config:
            create_params["InputDataConfig"] = input_data_config

        self.sagemaker.create_training_job(**create_params)
        print(f"Training job created: {job_name}")

    def _monitor_job(self, job_name: str) -> int:
        """Monitor training job and stream logs."""
        print(f"Monitoring job: {job_name}")

        log_group = "/aws/sagemaker/TrainingJobs"
        log_stream = None
        next_token = None

        while True:
            # Get job status
            response = self.sagemaker.describe_training_job(TrainingJobName=job_name)
            status = response["TrainingJobStatus"]

            # Try to find log stream
            if not log_stream:
                log_stream = self._find_log_stream(log_group, job_name)

            # Stream logs
            if log_stream:
                next_token = self._stream_logs(log_group, log_stream, next_token)

            # Check terminal states
            if status == "Completed":
                print(f"\nTraining job completed successfully")
                return 0
            elif status == "Failed":
                failure_reason = response.get("FailureReason", "Unknown")
                print(f"\nTraining job failed: {failure_reason}")
                return 1
            elif status == "Stopped":
                print(f"\nTraining job was stopped")
                return 1
            else:
                # InProgress, Starting, etc.
                time.sleep(10)

    def _find_log_stream(self, log_group: str, job_name: str) -> Optional[str]:
        """Find CloudWatch log stream for job."""
        try:
            response = self.logs.describe_log_streams(
                logGroupName=log_group,
                logStreamNamePrefix=job_name,
                orderBy="LogStreamName",
                limit=1,
            )
            streams = response.get("logStreams", [])
            if streams:
                return streams[0]["logStreamName"]
        except Exception:
            pass
        return None

    def _stream_logs(
        self, log_group: str, log_stream: str, next_token: Optional[str]
    ) -> Optional[str]:
        """Stream logs from CloudWatch."""
        try:
            params = {
                "logGroupName": log_group,
                "logStreamName": log_stream,
                "startFromHead": True,
            }
            if next_token:
                params["nextToken"] = next_token

            response = self.logs.get_log_events(**params)

            for event in response.get("events", []):
                print(event["message"])

            return response.get("nextForwardToken")
        except Exception:
            return next_token

    def _download_outputs(self, s3_path: str, job_name: str, local_dir: str):
        """Download training outputs from S3."""
        import boto3

        print(f"Downloading outputs to {local_dir}...")

        # Construct output path (SageMaker adds job_name/output/model.tar.gz)
        s3_output = f"{s3_path.rstrip('/')}/{job_name}/output/"

        # Use AWS CLI for recursive download (simpler than boto3)
        os.makedirs(local_dir, exist_ok=True)
        cmd = ["aws", "s3", "sync", s3_output, local_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Outputs downloaded to {local_dir}")
        else:
            print(f"Warning: Failed to download outputs: {result.stderr}")

    def _setup_cleanup_handlers(self):
        """Setup handlers for cleanup on exit/interrupt."""
        atexit.register(self._cleanup)

        def signal_handler(signum, frame):
            print("\nInterrupted! Stopping training job...")
            self._cleanup()
            sys.exit(130)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _cleanup(self):
        """Stop current training job if running."""
        if not self.current_job_name:
            return

        try:
            print(f"Stopping training job: {self.current_job_name}")
            self.sagemaker.stop_training_job(TrainingJobName=self.current_job_name)
            print("Stop request sent")
        except Exception as e:
            print(f"Warning: Failed to stop job: {e}")
        finally:
            self.current_job_name = None

    # =========================================================================
    # Job Management Methods (for cvl jobs/logs/kill commands)
    # =========================================================================

    def list_jobs(self, max_results: int = 20, status_filter: Optional[str] = None) -> List[dict]:
        """
        List recent SageMaker training jobs.

        Args:
            max_results: Maximum number of jobs to return
            status_filter: Filter by status (InProgress, Completed, Failed, Stopped)

        Returns:
            List of job info dicts with keys: job_id, status, created, duration
        """
        kwargs = {
            "MaxResults": max_results,
            "SortBy": "CreationTime",
            "SortOrder": "Descending",
        }
        if status_filter:
            kwargs["StatusEquals"] = status_filter

        response = self.sagemaker.list_training_jobs(**kwargs)
        jobs = []
        for job in response.get("TrainingJobSummaries", []):
            jobs.append({
                "job_id": job["TrainingJobName"],
                "status": job["TrainingJobStatus"],
                "created": job["CreationTime"].isoformat(),
                "duration": job.get("TrainingEndTime", job["CreationTime"]) - job["CreationTime"],
            })
        return jobs

    def get_job_status(self, job_id: str) -> dict:
        """
        Get detailed status of a training job.

        Args:
            job_id: SageMaker training job name

        Returns:
            Dict with status details
        """
        response = self.sagemaker.describe_training_job(TrainingJobName=job_id)
        return {
            "job_id": job_id,
            "status": response["TrainingJobStatus"],
            "secondary_status": response.get("SecondaryStatus"),
            "failure_reason": response.get("FailureReason"),
            "created": response["CreationTime"].isoformat(),
            "duration_seconds": response.get("TrainingTimeInSeconds"),
            "instance_type": response["ResourceConfig"]["InstanceType"],
            "output_path": response.get("OutputDataConfig", {}).get("S3OutputPath"),
        }

    def tail_logs(self, job_id: str, follow: bool = True):
        """
        Tail CloudWatch logs for a training job.

        Args:
            job_id: SageMaker training job name
            follow: If True, continuously poll for new logs
        """
        log_group = "/aws/sagemaker/TrainingJobs"

        # Find log stream
        try:
            streams = self.logs.describe_log_streams(
                logGroupName=log_group,
                logStreamNamePrefix=job_id,
                orderBy="LogStreamName",
                descending=True,
                limit=1,
            )
            if not streams.get("logStreams"):
                print(f"No logs found for job: {job_id}")
                return
            log_stream = streams["logStreams"][0]["logStreamName"]
        except Exception as e:
            print(f"Error finding log stream: {e}")
            return

        print(f"Tailing logs for: {job_id}")
        print("-" * 50)

        next_token = None
        while True:
            try:
                kwargs = {
                    "logGroupName": log_group,
                    "logStreamName": log_stream,
                    "startFromHead": True,
                }
                if next_token:
                    kwargs["nextToken"] = next_token

                response = self.logs.get_log_events(**kwargs)

                for event in response.get("events", []):
                    print(event["message"])

                new_token = response.get("nextForwardToken")

                if not follow:
                    break

                # Check if job is still running
                status = self.get_job_status(job_id)
                if status["status"] not in ("InProgress", "Stopping"):
                    # Print any remaining logs
                    if new_token != next_token:
                        next_token = new_token
                        continue
                    print("-" * 50)
                    print(f"Job {status['status']}")
                    break

                next_token = new_token
                time.sleep(5)

            except KeyboardInterrupt:
                print("\nStopped tailing logs")
                break
            except Exception as e:
                print(f"Error reading logs: {e}")
                break

    def stop_job(self, job_id: str) -> bool:
        """
        Stop a running training job.

        Args:
            job_id: SageMaker training job name

        Returns:
            True if stop request was sent successfully
        """
        try:
            self.sagemaker.stop_training_job(TrainingJobName=job_id)
            print(f"Stop request sent for: {job_id}")
            return True
        except self.sagemaker.exceptions.ClientError as e:
            if "ValidationException" in str(e):
                print(f"Job not running or already stopped: {job_id}")
            else:
                print(f"Error stopping job: {e}")
            return False
