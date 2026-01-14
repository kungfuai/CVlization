"""Job management commands for CVL CLI.

Commands:
- cvl jobs [--runner TYPE] [--status STATUS] - List jobs
- cvl logs <job-id> [--runner TYPE] [--no-follow] - Tail job logs
- cvl kill <job-id> [--runner TYPE] - Stop a running job
"""

from typing import Optional


def list_jobs(
    runner: Optional[str] = None,
    status: Optional[str] = None,
    max_results: int = 20,
    region: Optional[str] = None,
) -> int:
    """
    List jobs across runners.

    Args:
        runner: Filter by runner type (sagemaker, k8s, etc.)
        status: Filter by status (running, completed, failed)
        max_results: Maximum number of jobs to show
        region: AWS region for SageMaker

    Returns:
        Exit code
    """
    # For now, only SageMaker is implemented
    if runner and runner != "sagemaker":
        print(f"Runner '{runner}' job listing not yet implemented")
        print("Supported: sagemaker")
        return 1

    try:
        from cvl.runners import SageMakerRunner
    except ImportError:
        print("boto3 not installed. Install with: pip install boto3")
        return 1

    # Get role ARN from environment or use a placeholder
    import os
    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN", "")
    region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    if not role_arn:
        print("Note: SAGEMAKER_ROLE_ARN not set, using minimal client")
        # Create a minimal runner just for listing (doesn't need role for list)
        try:
            import boto3
            sagemaker = boto3.client("sagemaker", region_name=region)

            # Map status filter
            status_map = {
                "running": "InProgress",
                "completed": "Completed",
                "failed": "Failed",
                "stopped": "Stopped",
            }
            sm_status = status_map.get(status) if status else None

            kwargs = {
                "MaxResults": max_results,
                "SortBy": "CreationTime",
                "SortOrder": "Descending",
            }
            if sm_status:
                kwargs["StatusEquals"] = sm_status

            response = sagemaker.list_training_jobs(**kwargs)
            jobs = response.get("TrainingJobSummaries", [])

            if not jobs:
                print("No jobs found")
                return 0

            # Print header
            print(f"{'JOB ID':<45} {'STATUS':<12} {'DURATION':<10} {'CREATED'}")
            print("-" * 90)

            for job in jobs:
                job_id = job["TrainingJobName"]
                status = job["TrainingJobStatus"]
                created = job["CreationTime"].strftime("%Y-%m-%d %H:%M")

                end_time = job.get("TrainingEndTime", job["CreationTime"])
                duration = end_time - job["CreationTime"]
                duration_str = f"{int(duration.total_seconds())}s"

                print(f"{job_id:<45} {status:<12} {duration_str:<10} {created}")

            return 0

        except Exception as e:
            print(f"Error listing jobs: {e}")
            return 1
    else:
        # Use full runner
        runner_obj = SageMakerRunner(role_arn=role_arn, region=region)

        status_map = {
            "running": "InProgress",
            "completed": "Completed",
            "failed": "Failed",
            "stopped": "Stopped",
        }
        sm_status = status_map.get(status) if status else None

        jobs = runner_obj.list_jobs(max_results=max_results, status_filter=sm_status)

        if not jobs:
            print("No jobs found")
            return 0

        print(f"{'JOB ID':<45} {'STATUS':<12} {'DURATION':<10} {'CREATED'}")
        print("-" * 90)

        for job in jobs:
            duration_str = f"{job['duration_seconds']}s"
            created = job['created'][:16].replace('T', ' ')
            print(f"{job['job_id']:<45} {job['status']:<12} {duration_str:<10} {created}")

        return 0


def tail_logs(
    job_id: str,
    runner: Optional[str] = None,
    follow: bool = True,
    region: Optional[str] = None,
) -> int:
    """
    Tail logs for a job.

    Args:
        job_id: Job ID to tail logs for
        runner: Runner type (auto-detected if not specified)
        follow: Whether to follow logs (default: True)
        region: AWS region for SageMaker

    Returns:
        Exit code
    """
    # Auto-detect runner from job_id prefix
    if runner is None:
        if job_id.startswith("cvl-") and "-train-" in job_id:
            runner = "sagemaker"
        else:
            print(f"Cannot auto-detect runner for job: {job_id}")
            print("Specify runner with --runner (sagemaker, k8s, etc.)")
            return 1

    if runner != "sagemaker":
        print(f"Runner '{runner}' log tailing not yet implemented")
        return 1

    try:
        from cvl.runners import SageMakerRunner
    except ImportError:
        print("boto3 not installed. Install with: pip install boto3")
        return 1

    import os
    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN", "arn:aws:iam::000000000000:role/placeholder")
    region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    runner_obj = SageMakerRunner(role_arn=role_arn, region=region)
    runner_obj.tail_logs(job_id, follow=follow)

    return 0


def kill_job(
    job_id: str,
    runner: Optional[str] = None,
    region: Optional[str] = None,
) -> int:
    """
    Stop a running job.

    Args:
        job_id: Job ID to stop
        runner: Runner type (auto-detected if not specified)
        region: AWS region for SageMaker

    Returns:
        Exit code
    """
    # Auto-detect runner from job_id prefix
    if runner is None:
        if job_id.startswith("cvl-") and "-train-" in job_id:
            runner = "sagemaker"
        else:
            print(f"Cannot auto-detect runner for job: {job_id}")
            print("Specify runner with --runner (sagemaker, k8s, etc.)")
            return 1

    if runner != "sagemaker":
        print(f"Runner '{runner}' job stopping not yet implemented")
        return 1

    try:
        from cvl.runners import SageMakerRunner
    except ImportError:
        print("boto3 not installed. Install with: pip install boto3")
        return 1

    import os
    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN", "arn:aws:iam::000000000000:role/placeholder")
    region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    runner_obj = SageMakerRunner(role_arn=role_arn, region=region)
    success = runner_obj.stop_job(job_id)

    return 0 if success else 1
