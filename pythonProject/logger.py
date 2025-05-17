import csv
import os
from datetime import datetime

class CSVLogger:
    def __init__(self, source_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"detections_{source_name}_{timestamp}.csv"
        self.file = open(self.filename, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['timestamp', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2'])

    def log(self, detections, model_names, selected_classes=None):
        now = datetime.now().isoformat(timespec='seconds')
        for *xyxy, conf, cls in detections:
            class_name = model_names[int(cls)]
            if selected_classes is None or class_name.lower() in selected_classes:
                x1, y1, x2, y2 = map(int, xyxy)
                self.writer.writerow([now, class_name, round(float(conf), 2), x1, y1, x2, y2])

    def close(self):
        self.file.close()
