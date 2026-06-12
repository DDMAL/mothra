from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from auth_api import router as auth_router
from encode_api import router as encode_router
from account_api import router as account_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(encode_router)
app.include_router(account_router)