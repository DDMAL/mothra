from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

from auth_api import router as auth_router
from encode_api import router as encode_router
from account_api import router as account_router
from inference_api import router as inference_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","), 
    allow_methods=["*"], 
    allow_headers=["*"],
)
app.include_router(auth_router, prefix="/api")
app.include_router(encode_router, prefix="/api")
app.include_router(account_router, prefix="/api")
app.include_router(inference_router, prefix="/api")

_neon_dir = Path(__file__).parent.parent / "public" / "neon"
if _neon_dir.exists():
    app.mount("/neon", StaticFiles(directory=str(_neon_dir), html=True), name="neon")
_neon_gh_dir = _neon_dir / "Neon-gh"
if _neon_gh_dir.exists():
    app.mount("/Neon-gh", StaticFiles(directory=str(_neon_gh_dir)), name="neon-gh")

DIST_DIR = Path(__file__).parent.parent / "dist"
if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        return FileResponse(DIST_DIR / "index.html")