import cv2
import os

def get_video_source():
    print("Choose input source:")
    print("1. Webcam")
    print("2. Video file")
    choice = input("Enter 1 or 2: ")
    if choice == "1":
        return cv2.VideoCapture(0), "webcam"
    elif choice == "2":
        file_path = input("Enter full path to video file: ")
        if not os.path.exists(file_path):
            print("❌ File not found.")
            exit(1)
        return cv2.VideoCapture(file_path), os.path.splitext(os.path.basename(file_path))[0]
    else:
        print("❌ Invalid choice.")
        exit(1)