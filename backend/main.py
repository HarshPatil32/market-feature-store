from fastapi import FastAPI

from backend.api.router import router

app = FastAPI(title="market-feature-store")
app.include_router(router)
