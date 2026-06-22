from ultralytics import YOLO

# Tải mô hình YOLOv8 Segmentation
model = YOLO("yolov8n-seg.pt")

# Dự đoán hình ảnh và lưu ảnh về sẵn mặt nạ phân đoạn
results = model("xe01.jpg", save=True)

# Trích xuất dữ liệu mặt nạ
for r in results:
    masks = r.masks

    if masks is not None:
        # Lấy tọa độ các đa giác bao quanh đối tượng (dạng chuẩn hóa hoặc pixel)
        polygons = masks.xy  # Danh sách tọa độ [x, y] của đa giác

        for i, polygon in enumerate(polygons):
            cls = int(r.boxes.cls[i].item())
            name = model.names[cls]

            print(
                f"Đối tượng {name} thứ {i+1} có đa giác phân đoạn gồm "
                f"{len(polygon)} điểm tọa độ."
            )