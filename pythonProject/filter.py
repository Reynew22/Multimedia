import cv2

def select_classes(model):
    print("\nAvailable classes:")
    for i, name in model.names.items():
        print(f"{i}: {name}")

    user_input = input("🎯 Enter class names to detect (comma-separated), or press Enter for all: ")
    if user_input.strip():
        selected = set(name.strip().lower() for name in user_input.split(","))
        print(f"🔎 Filtering to: {', '.join(selected)}")
        return selected
    return None

def filter_detections(results, frame, model, selected_classes):
    if selected_classes:
        annotated = frame.copy()
        for *xyxy, conf, cls in results.xyxy[0]:
            class_name = model.names[int(cls)]
            if class_name.lower() in selected_classes:
                label = f"{class_name} {conf:.2f}"
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return annotated
    else:
        return results.render()[0]