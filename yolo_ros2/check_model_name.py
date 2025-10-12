from ultralytics import YOLO
model = YOLO('./models/yolo11n.pt')
print(model.names)
