from collections import defaultdict


class ObjectCounter:
    def __init__(self):
        # tracks total counts per frame
        self.total_counts = defaultdict(int)

    def update(self, detections, model_names, selected_classes=None):
        frame_counts = defaultdict(int)

        for *xyxy, conf, cls in detections:
            class_name = model_names[int(cls)]
            if selected_classes is None or class_name.lower() in selected_classes:
                frame_counts[class_name] += 1
                self.total_counts[class_name] += 1

        return frame_counts

    def get_total_counts(self):
        return dict(self.total_counts)
