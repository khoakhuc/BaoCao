from ultralytics import YOLO
from PIL import Image

# Tải mô hình YOLOv8 Object Detection (bản Nano siêu nhẹ)
model = YOLO("yolov8n.pt")

# Dự đoán hình ảnh và tự động lưu kết quả về sẵn khung nhận diện
results = model("xe01.jpg")

for r in results:
    print(r.boxes)
    im_array = r.plot() # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1]) # RGB PIL image
    im.show() # show image
    im.save('kq.jpg')