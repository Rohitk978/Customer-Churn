from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from src.api.routes import router


BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title = "Customer Churn Prediction API",
    description="Customer churn prediction application",
    version = "1.0.0"
)

app.include_router(router)

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name = "static"
)

@app.get("/", include_in_schema=False)
def home():
    """serve the main frontend application"""

    return FileResponse(FRONTEND_DIR/"index.html")

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )