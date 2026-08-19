from dotenv import load_dotenv
load_dotenv()

from celery import Celery
from config import CELERY_BROKER_URL

celery_app = Celery(
    "mothra",
    broker=CELERY_BROKER_URL,
    include=["tasks_predict", "tasks_encode", "tasks_text_batch", "tasks_cleanup"]
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
    # job_uploads/job_sessions used to only get swept once, at backend
    # startup (main.py) -- a long-lived pod that never restarted never got
    # swept again (mothra#220 row 28). Runs via the worker's embedded beat
    # scheduler (-B flag, see dev.sh/docker-compose.yml/k8s/worker.yaml)
    # rather than a separate beat Deployment: worker is pinned to
    # replicas=1 anyway (row 31 -- import-time init_db()/_migrate_db()), so
    # there's no risk yet of multiple beat schedulers double-firing this.
    # Revisit if/when row 31 unblocks replicas>1.
    beat_schedule={
        "cleanup-stale-uploads-and-sessions": {
            "task": "cleanup.run_periodic",
            "schedule": 3600.0,  # hourly
        },
    },
)