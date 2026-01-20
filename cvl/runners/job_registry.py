"""Job registry for tracking running and completed jobs.

Stores job metadata locally for SSH-based runners (no API to query).
Cloud runners (SageMaker, K8s, SkyPilot) can query their APIs directly,
but we still track them here for unified `cvl jobs` output.
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict


@dataclass
class JobRecord:
    """Record of a job run."""
    job_id: str
    runner: str  # ssh, rsync, sagemaker, k8s, skypilot
    status: str  # running, completed, failed, stopped
    start_time: float
    end_time: Optional[float] = None

    # Connection info (varies by runner)
    host: Optional[str] = None  # SSH/Rsync
    region: Optional[str] = None  # SageMaker
    namespace: Optional[str] = None  # K8s
    cluster: Optional[str] = None  # SkyPilot

    # Command info
    example: Optional[str] = None
    preset: Optional[str] = None


class JobRegistry:
    """
    Local registry for tracking jobs across runners.

    Stored at ~/.cvl/jobs.json
    """

    DEFAULT_PATH = Path.home() / ".cvl" / "jobs.json"

    def __init__(self, path: Optional[Path] = None):
        self.path = path or self.DEFAULT_PATH
        self._ensure_dir()

    def _ensure_dir(self):
        """Create directory if needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict[str, dict]:
        """Load jobs from file."""
        if not self.path.exists():
            return {}
        try:
            with open(self.path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _save(self, jobs: Dict[str, dict]):
        """Save jobs to file."""
        with open(self.path, "w") as f:
            json.dump(jobs, f, indent=2)

    def add(self, job: JobRecord):
        """Add or update a job record."""
        jobs = self._load()
        jobs[job.job_id] = asdict(job)
        self._save(jobs)

    def get(self, job_id: str) -> Optional[JobRecord]:
        """Get a job by ID."""
        jobs = self._load()
        if job_id in jobs:
            return JobRecord(**jobs[job_id])
        return None

    def update_status(self, job_id: str, status: str, end_time: Optional[float] = None):
        """Update job status."""
        jobs = self._load()
        if job_id in jobs:
            jobs[job_id]["status"] = status
            if end_time:
                jobs[job_id]["end_time"] = end_time
            self._save(jobs)

    def list_jobs(self, runner: Optional[str] = None, status: Optional[str] = None) -> List[JobRecord]:
        """List jobs, optionally filtered by runner or status."""
        jobs = self._load()
        results = []
        for job_data in jobs.values():
            if runner and job_data.get("runner") != runner:
                continue
            if status and job_data.get("status") != status:
                continue
            results.append(JobRecord(**job_data))
        # Sort by start_time descending (most recent first)
        results.sort(key=lambda j: j.start_time, reverse=True)
        return results

    def remove(self, job_id: str):
        """Remove a job record."""
        jobs = self._load()
        if job_id in jobs:
            del jobs[job_id]
            self._save(jobs)

    def cleanup_old(self, max_age_days: int = 7):
        """Remove completed/failed jobs older than max_age_days."""
        jobs = self._load()
        cutoff = time.time() - (max_age_days * 86400)
        to_remove = []
        for job_id, job_data in jobs.items():
            if job_data.get("status") in ("completed", "failed", "stopped"):
                if job_data.get("end_time", 0) < cutoff:
                    to_remove.append(job_id)
        for job_id in to_remove:
            del jobs[job_id]
        if to_remove:
            self._save(jobs)
        return len(to_remove)
