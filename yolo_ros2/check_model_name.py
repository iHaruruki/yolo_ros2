from ultralytics import YOLO
model = YOLO('./models/yolo11m.pt')
print(model.names)
