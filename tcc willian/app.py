from fastapi import FastAPI, File, UploadFile
from typing import List
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use ["http://localhost:3000"] for specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the path to the 'logs_eventos' folder
LOGS_FOLDER = "logs_eventos"

# Ensure the logs folder exists
if not os.path.exists(LOGS_FOLDER):
    os.makedirs(LOGS_FOLDER)

@app.get("/list-logs", response_model=List[str])
async def list_logs():
    """
    List all files in the 'logs_eventos' folder with their extensions.
    """
    try:
        files = [f for f in os.listdir(LOGS_FOLDER) if os.path.isfile(os.path.join(LOGS_FOLDER, f))]
        return files
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload-log")
async def upload_log(file: UploadFile = File(...)):
    """
    Upload a file to the 'logs_eventos' folder.
    """
    file_path = os.path.join(LOGS_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    return {"filename": file.filename}
