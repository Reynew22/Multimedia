import torch

def load_model():
    return torch.hub.load('yolov5', 'yolov5s', source='local')

def run_detection(model, frame):
    return model(frame)

