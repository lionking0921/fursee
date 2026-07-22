from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/last.pt")

# 开始训练
results = model.train(
    data="training.yaml",
    epochs=150,           			
    imgsz=1280,           			
    batch=8,            			
    device=0,             			
    project="runs/train", 			
    name="yolo26_stage1",   		
    save=True             			
)

print("Training completed. Best weights are saved to runs/train/yolo26_stage1/weights/best.pt")