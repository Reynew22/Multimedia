from fastapi import FastAPI, Body
from threading import Thread
from main import run_detection_system
from typing import List
import uvicorn

app = FastAPI()
detection_thread = None
detection_running = False
selected_classes = None
global_counts = {}

@app.get("/status")
def status():
    return {
        "detection_running": detection_running,
        "selected_classes": list(selected_classes) if selected_classes else "all",
        "total_counts": global_counts
    }

@app.get("/counts")
def counts():
    return global_counts

@app.post("/set_classes")
def set_classes(classes: List[str] = Body(...)):
    global selected_classes
    selected_classes = set([c.lower() for c in classes])
    return {"message": "Updated class filter", "selected_classes": list(selected_classes)}

@app.post("/start")
def start_detection():
    global detection_thread, detection_running, global_counts
    if detection_running:
        return {"message": "Detection already running."}

    def detection_wrapper():
        global global_counts, detection_running
        detection_running = True
        global_counts = run_detection_system(selected_classes)
        detection_running = False

    detection_thread = Thread(target=detection_wrapper)
    detection_thread.start()
    return {"message": "Detection started."}

@app.post("/stop")
def stop_detection():
    return {"message": "Stopping not implemented. Press ESC to stop manually for now."}
