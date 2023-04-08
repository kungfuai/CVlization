from torchvision.datasets import MNIST
from cvlization.torch.training_pipeline.image_gen.ebm.uva_energy.pipeline import TrainingPipeline

DATASET_PATH = "./data/raw/MNIST"

class MNISTDatasetBuilder:
    def training_dataset(self):
        return MNIST(root=DATASET_PATH, train=True, download=True)

    def validation_dataset(self):
        return MNIST(root=DATASET_PATH, train=False, download=True)


TrainingPipeline().fit(MNISTDatasetBuilder())