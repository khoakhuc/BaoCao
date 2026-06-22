from ultralytics import YOLO

# Tải mô hình YOLOv8 Object Detection (bản Nano siêu nhẹ)
model = YOLO("yolov8n.pt")

# Dự đoán hình ảnh và tự động lưu kết quả về sẵn khung nhận diện
results = model("xe02.jpg", save=True)

# Trích xuất dữ liệu chi tiết
for r in results:
    boxes = r.boxes
    for box in boxes:

        # Tọa độ hộp giới hạn
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()   # Độ tin cậy (0.0 -> 1.0)
        cls = int(box.cls[0].item())  # ID của lớp đối tượng

        name = model.names[cls]  # Tên lớp (ví dụ: 'person', 'car')

        print(
            f"Đối tượng: {name} | Độ tin cậy: {conf:.2f} | "
            f"Tọa độ: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})"
        )