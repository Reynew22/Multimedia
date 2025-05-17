import cv2
from detector import load_model, run_detection
from video_source import get_video_source
from recorder import init_writer
from filter import select_classes
from counter import ObjectCounter
from logger import CSVLogger
from heatmap import MotionHeatmap


def run_detection_system(class_filter=None, input_source=None, save_output=False):
    model = load_model()

    if input_source:
        cap = cv2.VideoCapture(input_source)
        input_name = "file"
    else:
        cap, input_name = get_video_source()

    writer = init_writer(cap, input_name) if save_output else None
    selected_classes = class_filter if class_filter else select_classes(model)

    counter = ObjectCounter()
    logger = CSVLogger(input_name)

    # read first frame for heatmap
    ret, frame_init = cap.read()
    if not ret:
        print("❌ Failed to read video input.")
        return {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    heatmap_gen = MotionHeatmap(frame_init.shape)

    # detection loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = run_detection(model, frame)
        detections = results.xyxy[0]

        frame_counts = counter.update(detections, model.names, selected_classes)
        logger.log(detections, model.names, selected_classes)

        # annotate
        annotated = frame.copy()
        for *xyxy, conf, cls in detections:
            class_name = model.names[int(cls)]
            if selected_classes is None or class_name.lower() in selected_classes:
                label = f"{class_name} {conf:.2f}"
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # frame counts
        y_offset = 30
        for cls, count in frame_counts.items():
            cv2.putText(annotated, f"{cls}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 20

        # Heatmap
        heatmap_overlay = heatmap_gen.update(frame)
        if heatmap_overlay is not None:
            annotated = cv2.addWeighted(annotated, 1.0, heatmap_overlay, 0.4, 0)

        cv2.imshow("YOLOv5 Detection + Heatmap", annotated)
        if writer:
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Clean
    cap.release()
    if writer:
        writer.release()
    logger.close()
    cv2.destroyAllWindows()

    print("\n📊 Total objects detected:")
    for cls, count in counter.get_total_counts().items():
        print(f" - {cls}: {count}")

    print("\n✅ Finished.")
    return counter.get_total_counts()


if __name__ == "__main__":
    run_detection_system()

