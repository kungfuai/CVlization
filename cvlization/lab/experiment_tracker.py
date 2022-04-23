from dataclasses import dataclass
import enum
from typing import Optional


class ExperimentTrackerProvider(str, enum.Enum):
    WANDB = "wandb"
    MLFLOW = "mlflow"


@dataclass
class ExperimentTracker:
    provider: Optional[ExperimentTrackerProvider] = None
    # experiment_name corresponds to 'project' in wandb
    experiment_name: Optional[str] = None
    # run_name corresponds to 'experiment name' in wandb.
    run_name: Optional[str] = None

    mlflow_uri: Optional[str] = None
    wandb_api_key: Optional[str] = None

    def setup(self):
        return self

    def log_params(self, params: dict):
        if self.provider == ExperimentTrackerProvider.WANDB:
            import wandb

            # TODO: pass in project name.
            wandb.init(project=self.experiment_name)
            if self.config.run_name:
                wandb.run.name = self.config.run_name

    def log_metrics(self, metrics: dict):
        pass

    def log_artifact(self, local_artifact_path: str, remote_artifact_dir: str = None):
        pass

    def watch(self, model):
        pass

    def autolog(self):
        pass
