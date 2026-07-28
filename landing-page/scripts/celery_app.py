from dotenv import load_dotenv
load_dotenv()

from celery import Celery
from config import CELERY_BROKER_URL

celery_app = Celery(
    "mothra",
    broker=CELERY_BROKER_URL,
    include=["tasks_predict", "tasks_encode", "tasks_text_batch"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)