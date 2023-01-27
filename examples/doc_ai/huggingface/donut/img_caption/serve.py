import logging
import time
import io

import PIL
from fastapi import FastAPI, status, File, UploadFile

from cvlization.lab.conceptual_captions_for_donut import ConceptualCaptionsDatasetBuilder
from cvlization.torch.training_pipeline.doc_ai.huggingface.donut.pipeline import Donut
from cvlization.torch.training_pipeline.doc_ai.huggingface.donut.model import DonutPredictionTask

LOGGER = logging.getLogger(__name__)

app = FastAPI()

logging.basicConfig(level=logging.INFO)


@app.on_event("startup")
async def _load_pl_model():
    global pl_model
    config = {
        "task": DonutPredictionTask.CAPTION,
        "max_length": ConceptualCaptionsDatasetBuilder.max_length,
        "task_start_token": ConceptualCaptionsDatasetBuilder.task_start_token,
        "image_height": ConceptualCaptionsDatasetBuilder.image_height,
        "image_width": ConceptualCaptionsDatasetBuilder.image_width,
        "ignore_id": ConceptualCaptionsDatasetBuilder.ignore_id,
    }
    pl_model = Donut(**config).eval()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health", status_code=status.HTTP_200_OK)
def healthcheck():
    return {"healthcheck": "Everything's OK!"}


@app.post("/v1/predict")
async def predict(file: UploadFile = File(...)):
    image = PIL.Image.open(io.BytesIO(await file.read()))
    start = time.perf_counter()
    prediction = pl_model.predict(image)
    return {
        "prediction": prediction.replace("<s_caption>", "").replace("</s_caption>", "").strip(),
        "time": {
            "total": time.perf_counter() - start,
        }
    }
