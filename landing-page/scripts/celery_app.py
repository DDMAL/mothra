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
    # Predict/text-batch task runtime varies wildly (one image vs. a whole
    # batch) -- prefetch_multiplier=1 stops one worker thread from hoarding
    # several tasks while another sits idle waiting for its next prefetch.
    worker_prefetch_multiplier=1,
    # Ack after the task finishes, not when it's received: if a worker dies
    # mid-task (OOM kill under low container memory, a native-lib crash --
    # both have happened, see CLAUDE.md), the task gets redelivered instead
    # of silently vanishing. Trade-off: a task that reliably crashes the
    # whole process on every attempt would redeliver and could crash-loop,
    # since nothing here sets a max-retry ceiling.
    task_acks_late=True,
)