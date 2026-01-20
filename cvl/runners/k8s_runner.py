"""Kubernetes Job runner for CVL examples.

Runs CVL examples as Kubernetes Jobs on any k8s cluster (EKS, GKE, AKS, local).

Requirements:
- kubernetes Python client: pip install kubernetes
- Valid kubeconfig or in-cluster config
- Container image accessible from the cluster
"""

import atexit
import signal
import sys
import time
from typing import Optional, List, Dict

# Lazy import kubernetes to avoid hard dependency
_k8s_client = None
_k8s_config = None


def _get_k8s():
    """Lazy load kubernetes client."""
    global _k8s_client, _k8s_config
    if _k8s_client is None:
        try:
            from kubernetes import client, config
            _k8s_client = client
            _k8s_config = config
        except ImportError:
            print("kubernetes not installed. Install with: pip install kubernetes")
            sys.exit(1)
    return _k8s_client, _k8s_config


class K8sRunner:
    """
    Run CVL examples as Kubernetes Jobs.

    This runner:
    1. Creates a Kubernetes Job with the specified container image
    2. Waits for the job to complete
    3. Streams logs from the pod
    4. Cleans up the job on completion or interrupt

    Requirements:
    - Valid kubeconfig (~/.kube/config) or in-cluster config
    - Container image accessible from the cluster (e.g., from ECR, GCR, DockerHub)

    Example:
        runner = K8sRunner(namespace="ml-training")
        runner.run(
            image="my-registry/nanogpt:latest",
            command=["python", "train.py", "--max_iters=1000"],
            gpu=1,
        )
    """

    def __init__(
        self,
        namespace: str = "default",
        kubeconfig: Optional[str] = None,
        context: Optional[str] = None,
    ):
        """
        Initialize Kubernetes runner.

        Args:
            namespace: Kubernetes namespace for jobs (default: "default")
            kubeconfig: Path to kubeconfig file (default: ~/.kube/config)
            context: Kubernetes context to use (default: current context)
        """
        client, config = _get_k8s()

        # Load config
        try:
            if kubeconfig:
                config.load_kube_config(config_file=kubeconfig, context=context)
            else:
                try:
                    config.load_incluster_config()
                except config.ConfigException:
                    config.load_kube_config(context=context)
        except Exception as e:
            raise RuntimeError(f"Failed to load Kubernetes config: {e}")

        self.namespace = namespace
        self.batch_v1 = client.BatchV1Api()
        self.core_v1 = client.CoreV1Api()

        # Track current job for cleanup
        self.current_job_name = None
        self._setup_cleanup_handlers()

    def run(
        self,
        image: str,
        command: List[str],
        name_prefix: str = "cvl-job",
        gpu: int = 0,
        cpu: str = "1",
        memory: str = "4Gi",
        timeout_minutes: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        node_selector: Optional[Dict[str, str]] = None,
        image_pull_secrets: Optional[List[str]] = None,
        working_dir: Optional[str] = None,
        delete_on_completion: bool = True,
    ) -> int:
        """
        Run a command as a Kubernetes Job.

        Args:
            image: Container image to run
            command: Command and arguments to run
            name_prefix: Prefix for job name (default: "cvl-job")

            gpu: Number of GPUs to request (default: 0)
            cpu: CPU request/limit (default: "1")
            memory: Memory request/limit (default: "4Gi")
            timeout_minutes: Job timeout in minutes (default: None)

            env: Environment variables dict
            node_selector: Node selector labels (e.g., {"gpu": "true"})
            image_pull_secrets: List of secret names for private registries
            working_dir: Working directory in container
            delete_on_completion: Delete job after completion (default: True)

        Returns:
            Exit code (0 for success)
        """
        client, _ = _get_k8s()

        # Generate unique job name
        job_name = f"{name_prefix}-{int(time.time())}"
        self.current_job_name = job_name

        try:
            # Build job spec
            job = self._build_job(
                name=job_name,
                image=image,
                command=command,
                gpu=gpu,
                cpu=cpu,
                memory=memory,
                timeout_minutes=timeout_minutes,
                env=env,
                node_selector=node_selector,
                image_pull_secrets=image_pull_secrets,
                working_dir=working_dir,
            )

            # Create job
            print(f"Creating job: {job_name}")
            self.batch_v1.create_namespaced_job(namespace=self.namespace, body=job)
            print(f"Job created in namespace: {self.namespace}")

            # Wait for pod to be scheduled and stream logs
            exit_code = self._wait_and_stream_logs(job_name)

            return exit_code

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            return 130

        finally:
            if delete_on_completion and self.current_job_name:
                self._delete_job(self.current_job_name)
            self.current_job_name = None

    def _build_job(
        self,
        name: str,
        image: str,
        command: List[str],
        gpu: int,
        cpu: str,
        memory: str,
        timeout_minutes: Optional[int],
        env: Optional[Dict[str, str]],
        node_selector: Optional[Dict[str, str]],
        image_pull_secrets: Optional[List[str]],
        working_dir: Optional[str],
    ):
        """Build Kubernetes Job object."""
        client, _ = _get_k8s()

        # Resource requests/limits
        resources = {
            "requests": {"cpu": cpu, "memory": memory},
            "limits": {"cpu": cpu, "memory": memory},
        }
        if gpu > 0:
            resources["requests"]["nvidia.com/gpu"] = str(gpu)
            resources["limits"]["nvidia.com/gpu"] = str(gpu)

        # Environment variables
        env_vars = []
        if env:
            for k, v in env.items():
                env_vars.append(client.V1EnvVar(name=k, value=v))

        # Container spec
        container = client.V1Container(
            name="main",
            image=image,
            command=command[:1] if command else None,
            args=command[1:] if len(command) > 1 else None,
            resources=client.V1ResourceRequirements(**resources),
            env=env_vars or None,
            working_dir=working_dir,
            image_pull_policy="Always",
        )

        # Pod spec
        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy="Never",
            node_selector=node_selector,
        )

        # Image pull secrets
        if image_pull_secrets:
            pod_spec.image_pull_secrets = [
                client.V1LocalObjectReference(name=s) for s in image_pull_secrets
            ]

        # Job spec
        job_spec = client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={"job-name": name}),
                spec=pod_spec,
            ),
            backoff_limit=0,  # No retries
            ttl_seconds_after_finished=300,  # Auto-cleanup after 5 min
        )

        if timeout_minutes:
            job_spec.active_deadline_seconds = timeout_minutes * 60

        # Job object
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=name),
            spec=job_spec,
        )

        return job

    def _wait_and_stream_logs(self, job_name: str) -> int:
        """Wait for job pod and stream logs."""
        client, _ = _get_k8s()

        # Wait for pod to be created
        print("Waiting for pod to be scheduled...")
        pod_name = None
        for _ in range(60):  # 5 min timeout for scheduling
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"job-name={job_name}",
            )
            if pods.items:
                pod_name = pods.items[0].metadata.name
                pod_phase = pods.items[0].status.phase
                if pod_phase in ("Running", "Succeeded", "Failed"):
                    break
                print(f"  Pod status: {pod_phase}")
            time.sleep(5)

        if not pod_name:
            print("Error: Pod not created within timeout")
            return 1

        print(f"Pod: {pod_name}")
        print("-" * 50)

        # Stream logs
        try:
            for line in self.core_v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=self.namespace,
                follow=True,
                _preload_content=False,
            ).stream():
                print(line.decode("utf-8"), end="")
        except client.exceptions.ApiException as e:
            if e.status != 404:  # Ignore if pod already gone
                print(f"Log streaming error: {e}")

        print("-" * 50)

        # Get final status
        job = self.batch_v1.read_namespaced_job(name=job_name, namespace=self.namespace)

        if job.status.succeeded:
            print(f"Job completed successfully")
            return 0
        elif job.status.failed:
            print(f"Job failed")
            return 1
        else:
            print(f"Job status unknown: {job.status}")
            return 1

    def _delete_job(self, job_name: str):
        """Delete job and its pods."""
        client, _ = _get_k8s()
        try:
            print(f"Deleting job: {job_name}")
            self.batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(propagation_policy="Foreground"),
            )
        except client.exceptions.ApiException as e:
            if e.status != 404:
                print(f"Warning: Failed to delete job: {e}")

    def _setup_cleanup_handlers(self):
        """Setup handlers to clean up job on exit."""
        def cleanup():
            if self.current_job_name:
                self._delete_job(self.current_job_name)

        atexit.register(cleanup)

        def signal_handler(signum, frame):
            cleanup()
            sys.exit(130)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
