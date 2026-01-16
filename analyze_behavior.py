from src.behavior_analyzer import analyze_video

if __name__ == "__main__":
    analyze_video(
        video_path="data_test/test_videos/normal_case.mp4",
        yolo_w="models/best.pt",
        gaze_onnx="models/l2cs_gaze.onnx",
        headpose_onnx="models/sixdrepnet.onnx"
    )
