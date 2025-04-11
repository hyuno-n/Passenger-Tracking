from ultralytics import YOLO

# 🔧 모델 로드 (기존 모델 사용)
model = YOLO("head.pt")

# 🔁 학습 시작
model.train(
    data="dataset/head_data.yaml",  # 데이터셋 yaml 경로
    epochs=100,                      # 에폭 수
    batch=16,                       # 배치 크기
    imgsz=640,                      # 입력 이미지 크기
    name="head_finetuned",         # 저장 폴더 이름
    lr0=0.001,                      # 초기 학습률
    workers=4,                      # 데이터 로딩 워커 수
    device=0                        # GPU ID (CPU는 'cpu')
)

print("✅ 학습 완료! 결과: runs/detect/head_finetuned/")