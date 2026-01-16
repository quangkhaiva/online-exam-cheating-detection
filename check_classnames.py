from ultralytics import YOLO

# chọn model cần kiểm tra
model_path = "models/best_v11.pt"   # hoặc "models/best.pt"

# load model
model = YOLO(model_path)

# in ra danh sách class của model
print("🔍 Class mapping của model:")
for k, v in model.names.items():
    print(f"{k}: {v}")
