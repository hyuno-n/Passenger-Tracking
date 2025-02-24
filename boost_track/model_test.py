import torch
import numpy as np
import torchvision.transforms as T
from tracker.embedding import ModelFactory, ModelConfigFactory

WEIGHTS_DIR = "external/weights"

WEIGHTS = {
    'swinv2': f'{WEIGHTS_DIR}/Micrsorft_swinv2_large_patch4_window12_192_22k.pth',
    'convNext': f'{WEIGHTS_DIR}/convnext_xlarge_22k_1k_384_ema.pth',
    'La_Transformer': f'{WEIGHTS_DIR}/LaTransformer.pth',
    'VIT': f'{WEIGHTS_DIR}/vit_base_ics_cfs_lup.pth',
    'dinov2': f'{WEIGHTS_DIR}/dinov2_vitb14_pretrain.pth',
    'DETR': f'{WEIGHTS_DIR}/r50_deformable_detr.pth',
}

def test_model_preprocessing_and_output(model_type):
    """
    모델별 입력 전처리 및 출력값 형태를 자동으로 확인하는 테스트 함수

    :param model_name: 테스트할 모델 이름 (ex: 'dinov2', 'CLIP', 'swinv2', ...)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 설정 불러오기
    config = ModelConfigFactory.create_config(model_type)
    config.reid_model_path = WEIGHTS[model_type]  # 가중치 경로 추가

    print(f"🔍 모델명: {config.model_type}, 가중치 경로: {config.reid_model_path}")  # 디버깅용 출력

    model = ModelFactory.create_model(config, device)
    # 모델 생성
    try:
        model = ModelFactory.create_model(config, device)
        model.eval()
    except Exception as e:
        print(f"[ERROR] 모델 {model_name} 로드 실패: {e}")
        return

    print(f"\n🚀 테스트 모델: {model_name}")
    print(f"📌 모델 입력 크기: {config.crop_size}")

    # 더미 이미지 생성 (기본적으로 흰색 이미지)
    dummy_image = np.ones((config.crop_size[1], config.crop_size[0], 3), dtype=np.uint8) * 255
    
    # 전처리 수행
    transform = config.get_transform()
    try:
        preprocessed_tensor = transform(dummy_image).unsqueeze(0).to(device)
        print(f"✅ 전처리 성공! 입력 Tensor 형태: {preprocessed_tensor.shape}")
    except Exception as e:
        print(f"[ERROR] 전처리 실패: {e}")
        return

    # 모델 실행 및 출력값 확인
    try:
        with torch.no_grad():
            output = model(preprocessed_tensor)
        
        if isinstance(output, dict):
            for key, value in output.items():
                if isinstance(value, torch.Tensor):  # 🔍 리스트가 아닌 경우에만 shape 출력
                    print(f"🔍 출력 Key: {key}, Shape: {value.shape}")
                else:
                    print(f"⚠️ 출력 Key: {key} 는 리스트 또는 다른 형식: {(value)}")
        elif isinstance(output, torch.Tensor):
            print(f"✅ 모델 출력 Tensor 형태: {output.shape}")
        else:
            print(f"⚠️ 예상치 못한 출력 타입: {type(output)}")

    except Exception as e:
        print(f"[ERROR] 모델 실행 실패: {e}")

# 테스트할 모델 목록
model_list = ["DETR"]

# 모든 모델에 대해 테스트 실행
for model_name in model_list:
    test_model_preprocessing_and_output(model_name)
