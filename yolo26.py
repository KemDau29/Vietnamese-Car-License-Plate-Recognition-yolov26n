from ultralytics import YOLO

#model = YOLO("yolo26n.pt")  
model = YOLO("D:/HCMUTE/DIGITAL IMAGE PROCESSING/yolo26/best.pt")

# model.train(
#     data='/dataset/data.yaml',
#     epochs=50,
#     imgsz=640,
#     device='cpu'
# )
result = model('car2.jpg')

for r in result:
    r.show()
        