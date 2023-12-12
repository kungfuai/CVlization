from cvlization.torch.training_pipeline.seem.pipeline.XDecoderPipeline import XDecoderPipeline


REQUIREMENTS = """
infinibatch>=0.1.1
git+https://github.com/MaureenZOU/detectron2-xyz.git
"""


def main():
    prediction_pipeline = XDecoderPipeline()


if __name__ == "__main__":
    main()