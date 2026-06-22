from ultralytics import YOLO

# Tải mô hình YOLOv8 Classification (ví dụ: yolov8n-cls.pt)
model = YOLO("yolov8n-cls.pt")

# Thực hiện phân loại hình ảnh
results = model("xe01.jpg")

# Xem kết quả phân loại
for r in results:
    probs = r.probs

    top1_idx = probs.top1              # ID lớp có xác suất cao nhất
    top1_conf = probs.top1conf.item()  # Xác suất của lớp đó
    top1_name = model.names[top1_idx]  # Tên lớp

    print(f"Nhãn dự đoán cao nhất: {top1_name} ({top1_conf * 100:.2f}%)")

    # Lấy top 5 dự đoán tốt nhất
    print("Top 5 dự đoán:")
    top5_indices = probs.top5
    top5_confs = probs.top5conf

    for idx, conf in zip(top5_indices, top5_confs):
        print(f" - {model.names[idx]}: {conf.item() * 100:.2f}%")