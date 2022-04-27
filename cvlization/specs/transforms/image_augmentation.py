from abc import abstractmethod


class ImageAugmentation:
    @abstractmethod
    def transform(self, image, target):
        raise NotImplementedError
