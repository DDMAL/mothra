import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from auth_api import limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from auth_api import router as auth_router
from projects_api import router as projects_router
from images_api import router as images_router
from mei_api import router as mei_router
from models_api import router as models_router
from encode_api import router as encode_router
from account_api import router as account_router
from inference_api import router as inference_router
from ic_api import router as ic_router
from text_api import router as text_router
from cantus_api import router as cantus_router
from batch_api import router as batch_router
from jobs_api import router as jobs_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","), 
    allow_methods=["*"], 
    allow_headers=["*"],
)
app.include_router(auth_router, prefix="/api")
app.include_router(projects_router, prefix="/api")
app.include_router(images_router, prefix="/api")
app.include_router(mei_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(encode_router, prefix="/api")
app.include_router(account_router, prefix="/api")
app.include_router(inference_router, prefix="/api")
app.include_router(ic_router, prefix="/api")
app.include_router(text_router, prefix="/api")
app.include_router(cantus_router, prefix="/api")
app.include_router(batch_router, prefix="/api")
app.include_router(jobs_router, prefix="/api")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_neon_dir = Path(__file__).parent.parent / "public" / "neon"
if _neon_dir.exists():
    app.mount("/neon", StaticFiles(directory=str(_neon_dir), html=True), name="neon")
_neon_gh_dir = _neon_dir / "Neon-gh"
if _neon_gh_dir.exists():
    app.mount("/Neon-gh", StaticFiles(directory=str(_neon_gh_dir)), name="neon-gh")

DIST_DIR = Path(__file__).parent.parent / "dist"
# Guard on the actual build artifacts, not just dist/ — a partial dist/ (e.g.
# holding only the Neon submodule build, with no assets/ or index.html) exists
# in dev and would otherwise crash StaticFiles() at import. In dev the frontend
# is served by Vite on :5173, so the backend simply skips this mount.
if (DIST_DIR / "assets").is_dir() and (DIST_DIR / "index.html").is_file():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        return FileResponse(DIST_DIR / "index.html")