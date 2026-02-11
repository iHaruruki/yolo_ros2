from ultralytics import YOLO
model = YOLO('./models/yolo26n.pt')
print(model.names)
