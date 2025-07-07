
# BRT Seat Occupancy Detection - Refactored Codebase

이 프로젝트는 **BRT(간선급행버스)** 차량 내부에서 좌석 점유 여부를 판단하기 위한 컴퓨터 비전 기반 시스템입니다.  
어안 카메라 영상 → 평면화 → 사람 검출 → Homography 변환 → 좌석 포함 여부 판단이라는 파이프라인으로 구성되어 있으며,  
PyTorch 기반 YOLO 모델과 OpenCV, PyQt, NumPy 등 다양한 도구가 활용됩니다.

---

## 📁 디렉토리 구조 및 역할

### `core/`
핵심 로직 모듈. 여러 스크립트에서 공통으로 사용되는 함수들을 정리합니다.
- `projection.py` : 어안 영상 평면화, 다중 뷰 분리
- `homography_solver.py` : 좌표 기반 Homography 행렬 계산

### `utils/`
라벨, 통계 계산, 경로 처리 등 보조 함수들을 정리합니다.
- `label_utils.py` : 좌석 점유 라벨 비율 계산 등

### `scripts/`
학습, 테스트, 데이터셋 구성 등 직접 실행되는 스크립트입니다.
- `train_model.py` : YOLO 학습 실행
- `test_predict.py` : YOLO 검출 테스트
- `prepare_dataset.py` : 이미지 및 라벨 정리

### `tools/`
디버깅, 수작업 확인용 GUI/도구 모음입니다.
- `split_seat_gui.py` : 좌석 분할 수작업 도구
- `fisheye_grid_viewer.py` : 어안 시각화 확인
- `homography_demo.py` : 한 프레임 대상 Homography 적용 시각화

### `eval/`
정량적 성능 평가 및 좌석 점유 판단 관련 코드입니다.
- `eval_homography.py` : 시나리오 전체 평가 스크립트
- `check_occupancy.py` : Homography + 좌석 포함 판단
- `label_statistics.py` : 라벨 점유율, 통계 계산
- `split_checker.py` : 시나리오별 좌석 결과 비교

### `viz/`
좌석 결과 시각화, 멀티뷰 투영 결과 렌더링
- `seat_viewer.py`, `multiview_seat_visualizer.py`

### `data/`
전처리 관련 데이터 정리
- `extract_valid_frames.py` : 유효 프레임 필터링
- `prepare_bus_dataset.py` : BRT 데이터셋 구성

### `gui/`
PyQt 기반의 전체 좌석 점유 판단 GUI 툴
- `valid_gui.py` : 사용자 판단 수집을 위한 인터페이스

---