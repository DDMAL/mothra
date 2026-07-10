"""Cantus source lookup — proxies to the standalone text-service, which
wraps mothra-text's fetch_cantus_csv. Mirrors text_api.py's server-to-server
HTTP pattern: stdlib urllib, TEXT_API_URL env var, no extra dependency.
"""
from __future__ import annotations
import json
import os
import urllib.error
import urllib.request

from fastapi import APIRouter, Depends, HTTPException

from auth_api import get_current_user

router = APIRouter()

TEXT_API_URL = os.environ.get("TEXT_API_URL", "http://localhost:8002").rstrip("/")

@router.get("/cantus/source/{source_id}")
def get_cantus_source(source_id: int, user=Depends(get_current_user)):
    try:
        with urllib.request.urlopen(f"{TEXT_API_URL}/cantus-source/{source_id}", timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="ignore")
        try:
            detail = json.loads(detail).get("detail", detail)
        except json.JSONDecodeError:
            pass
        raise HTTPException(status_code=exc.code, detail=detail) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"text-service at {TEXT_API_URL} is unreachable: {exc}") from exc