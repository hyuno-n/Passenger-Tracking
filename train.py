from ultralytics import YOLO

# 기존 pretrained head 모델 로드
model = YOLO("head.pt")

# Freeze 전략 적용 학습
model.train(
    data="bus_dataset/head_data.yaml",
    epochs=100,             
    batch=16,
    imgsz=640,
    lr0=0.0005,             # 느리게 fine-tune
    name="add_tire_finetune_freeze",  # 모델 이름
    freeze=10,              # 앞쪽 10개 레이어 freeze
    patience=20,            # early stopping
    cos_lr=True,            # cosine decay
    device=0                # GPU 사용
)

print("✅ Freeze 기반 학습 완료! 모델: runs/detect/add_tire_finetune_freeze/weights/best.pt")
