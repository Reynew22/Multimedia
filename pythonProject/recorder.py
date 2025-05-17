import cv2
import os

def init_writer(cap, input_name):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = f"{input_name}_yolo_output.avi"
    print(f"💾 Output will be saved to: {output_path}")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
