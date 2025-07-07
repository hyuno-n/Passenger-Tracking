from ultralytics import YOLO

# 기존 pretrained head 모델 로드
model = YOLO("head.pt")

# Freeze 전략 적용 학습
model.train(
    data="bus_dataset/head_data.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    lr0=0.0005,
    name="head_finetuned_fpboost",
    patience=20,
    cos_lr=True,
    device=0  # or "cuda:0"
)

print("✅ Freeze 기반 학습 완료! 모델: runs/detect/finetune_freeze/weights/best.pt")
