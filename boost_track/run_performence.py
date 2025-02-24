from optparse import NO_DEFAULT
import os
import cv2
import numpy as np
import torch
import argparse
from collections import deque
from tqdm import tqdm
from dataclasses import dataclass
from ultralytics import YOLO
from natsort import natsorted
from default_settings import GeneralSettings, BoostTrackConfig
from tracker.boost_track import BoostTrack
import utils
from typing import Dict, List, Tuple , Optional
import random
from collections import deque
from evaluate import compute_mot_metrics, get_final_mot_metrics
import json

@dataclass
class VisualizationConfig:
    visualize: bool = False
    save_video: bool = True
    save_frame: bool = False
    stop_frame_ids: List[int] = None

class ModelConfig:
    WEIGHTS = {
        'swinv2': 'Micrsorft_swinv2_large_patch4_window12_192_22k.pth',
        'convNext': 'convnext_xlarge_22k_1k_384_ema.pth',
        'La_Transformer': 'LaTransformer.pth',
        'VIT': 'vit_base_ics_cfs_lup.pth',
        'dinov2': 'dinov2_vitb14_pretrain.pth'
    }

    @staticmethod
    def get_model_name(reid_model: str) -> str:
        model_name = os.path.splitext(reid_model)[0]
        return model_name

class Visualizer:
    def __init__(self):
        self.id_colors: Dict[int, List[int]] = {}
        
    def get_id_color(self, track_id: int) -> List[int]:
        if track_id not in self.id_colors:
            self.id_colors[track_id] = [random.randint(150, 255) for _ in range(3)]
        return self.id_colors[track_id]

    def draw_detection_boxes(self, image: np.ndarray, dets: np.ndarray, frame_mapping: Dict = None) -> np.ndarray:
        vis_img = image.copy()

        box_mapping = {}
        for i, (x1, y1, x2, y2, conf) in enumerate(dets): 
            if frame_mapping is not None and i in frame_mapping:
                track_id = frame_mapping[i]
                color = self.get_id_color(track_id)
                
                box_mapping.update({i: [track_id, x1, y1, x2, y2 , conf]})
                
                cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(vis_img, f"Y#{i}||T#{track_id}", (int(x1), int(y1)-10),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            else:
                cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
                cv2.putText(vis_img, f"Y#{i}", (int(x1), int(y1)-10),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        return vis_img , box_mapping

    def draw_tracking_results(self, image: np.ndarray, tlwhs: List, ids: List[int]) -> Tuple[np.ndarray, List[int]]:
        vis_img = image.copy()
        track_id_list = []
        for tlwh, track_id in zip(tlwhs, ids):
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            color = self.get_id_color(track_id)
            
            mid_x , mid_y = x1 + w/2 , y1 + h/2
            
            if track_id not in track_id_list:
                track_id_list.append(int(track_id))
                
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(vis_img, f"ID: {track_id}", (int(mid_x), int(mid_y)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
                        
        return vis_img, track_id_list

    def draw_xml_boxes(self, image: np.ndarray, xml_boxes: List , idx : int) -> np.ndarray:
        """XML 파일의 박스와 ID를 시각화"""
        vis_img = image.copy()
        for name, xmin, ymin, xmax, ymax in xml_boxes:
            try:
                id_num = int(''.join(filter(str.isdigit, name)))
                color = self.get_id_color(id_num)
                
                cv2.rectangle(vis_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(vis_img, f"ID: {id_num}", (int(xmin), int(ymin)-10),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            except:
                cv2.rectangle(vis_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 255), 2)
                cv2.putText(vis_img, name, (int(xmin), int(ymin)-10),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(vis_img, f"Frame: {idx}", (10, 30),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        return vis_img 

    def visualize_tracker_xml_mapping(self, image: np.ndarray, tracker_results: np.ndarray, xml_results: np.ndarray, tracker_label_mapping: dict) -> np.ndarray:
        """
        Tracker 박스와 XML 박스의 매핑 관계를 시각화합니다.

        Args:
            image: 원본 이미지
            tracker_results: Tracker 탐지 결과 박스 리스트 [[x1, y1, x2, y2, track_id, conf], ...]
            xml_file: XML 파일 경로
            tracker_label_mapping: Tracker ID와 XML 레이블 매핑 딕셔너리 {track_id: xml_label}
        
        Returns:
            np.ndarray: 매핑 관계가 시각화된 이미지
        """
        vis_img = image.copy()
        xml_boxes = xml_results  # XML GT 박스 불러오기

        # 색상 맵 생성 (매칭된 박스마다 고유한 색상 사용)
        colors = {}
        for xml_label in set(tracker_label_mapping.values()):
            colors[xml_label] = self.get_id_color(xml_label)
        # Tracker 박스 그리기 (실선)
        for track in tracker_results:
            x1, y1, x2, y2, track_id, _ = map(int, track)
            if track_id in tracker_label_mapping:
                xml_label = tracker_label_mapping[track_id]
                color = colors[xml_label]
                
                # Tracker 박스 (실선)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                
                # Tracker ID와 매칭된 XML 레이블 표시
                label = f"Track_{track_id}: {xml_label}"
                cv2.putText(vis_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 매칭된 XML 박스 찾기
                for xml_box in xml_boxes:
                    xml_name = int(xml_box[0])
                    if xml_name == xml_label:
                        xml_x1, xml_y1, xml_x2, xml_y2 = map(int, xml_box[1:])
                        # XML 박스 (점선)
                        for i in range(xml_x1, xml_x2, 10):
                            cv2.line(vis_img, (i, xml_y1), (i+5, xml_y1), color, 2)
                            cv2.line(vis_img, (i, xml_y2), (i+5, xml_y2), color, 2)
                        for i in range(xml_y1, xml_y2, 10):
                            cv2.line(vis_img, (xml_x1, i), (xml_x1, i+5), color, 2)
                            cv2.line(vis_img, (xml_x2, i), (xml_x2, i+5), color, 2)

                        # XML 박스 레이블 표시
                        label = f"XML: {xml_name}"
                        cv2.putText(vis_img, label, (xml_x1, xml_y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis_img

class VideoWriter:
    def __init__(self, save_dir: str, model_name: str, emb_method: str, video_name: str):
        os.makedirs(save_dir, exist_ok=True)
        name = f"{model_name}_{emb_method}_{video_name}"
        self.video_path = os.path.join(save_dir, f"{name}_tracking.mp4")
        self.writer = None

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                self.video_path,
                fourcc,
                15.0,
                (frame.shape[1], frame.shape[0]),
                True
            )
        self.writer.write(frame)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            print(f"Video saved to: {self.video_path}")

def save_metrics(model_name, final_mota, final_hota, final_idf1, total_fp, total_fn, total_idsw):
    results = {
        "Model": model_name,
        "MOTA": final_mota,
        "HOTA": final_hota,
        "IDF1": final_idf1,
        "False Positives": total_fp,
        "False Negatives": total_fn,
        "ID Switches": total_idsw
    }
    # 결과 저장
    save_path = f"results/{model_name}_metrics.json"
    os.makedirs('results', exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"[INFO] Saved metrics to {save_path}")

def is_mostly_inside(inner, outer, area_ratio_threshold=0.9):
    """작은 박스의 90% 이상이 큰 박스 안에 있는지 확인"""
    # 교집합 영역 계산
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 작은 박스의 전체 영역
    area_inner = (inner[2]-inner[0])*(inner[3]-inner[1])
    
    # 영역 비율 계산
    ratio = intersection / area_inner
    return ratio >= area_ratio_threshold

def remove_nested_boxes(boxes, scores):
    """중첩 박스 제거 알고리즘"""
    # 1. confidence score 기준으로 내림차순 정렬
    sorted_indices = np.argsort(scores)[::-1]  # 높은 점수부터 처리
    keep = []  # 유지할 박스의 인덱스
    
    while sorted_indices.size > 0:
        # 2. 가장 높은 confidence를 가진 박스 선택
        current_idx = sorted_indices[0]
        keep.append(current_idx)
        current_box = boxes[current_idx]
        
        # 3. 남은 박스들과 비교
        remaining = []
        for idx in sorted_indices[1:]:
            target_box = boxes[idx]
            
            # 4. 박스 크기 비교 (큰 박스 vs 작은 박스)
            area_current = (current_box[2]-current_box[0])*(current_box[3]-current_box[1])
            area_target = (target_box[2]-target_box[0])*(target_box[3]-target_box[1])
            
            # 5. 큰 박스와 작은 박스 관계 결정
            if area_current > area_target:
                large_box, small_box = current_box, target_box
            else:
                large_box, small_box = target_box, current_box
            
            # 6. 중첩 판단
            condition = is_mostly_inside(small_box, large_box)

            # 7. 중첩되지 않은 박스만 남김
            if not condition:
                remaining.append(idx)
        
        # 8. 다음 반복을 위해 남은 박스들로 업데이트
        sorted_indices = np.array(remaining)
    
    return [boxes[i] for i in keep], [scores[i] for i in keep]

def process_yolo_detection(results) -> Optional[np.ndarray]:
    """YOLO 탐지 결과를 후처리하여 중첩된 박스를 제거하고 필터링된 탐지 결과를 반환합니다.
    
    Args:
        results: YOLO 모델의 탐지 결과
        
    Returns:
        Optional[np.ndarray]: 필터링된 탐지 결과 배열 [x1, y1, x2, y2, confidence]
                             탐지된 객체가 없는 경우 None 반환
    """
    def extract_box_info(boxes: torch.Tensor) -> List[List[float]]:
        """YOLO 박스 정보를 추출하여 리스트로 변환"""
        detections = []
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            conf = boxes.conf[i].cpu().numpy()
            detections.append([*box, conf])
        return detections

    # 모든 탐지 결과 수집
    all_detections = []
    for result in results:
        all_detections.extend(extract_box_info(result.boxes))
    
    if not all_detections:
        return None
    
    # numpy 배열로 변환
    detections = np.array(all_detections)
    boxes = detections[:, :4]
    scores = detections[:, 4]
    
    # 중첩 박스 제거
    filtered_boxes, filtered_scores = remove_nested_boxes(boxes, scores)
    
    # 최종 결과 형식으로 변환
    filtered_detections = np.column_stack([
        filtered_boxes,
        np.array(filtered_scores).reshape(-1, 1)
    ])
    
    return filtered_detections

def setup_tracker(args) -> BoostTrack:
    config = BoostTrackConfig(
        reid_model_path=f'external/weights/{args.reid_model}',
        device='cuda',
        max_age=15,
        min_hits=0,
        det_thresh=0.55,  # 탐지 신뢰도 임계값
        iou_threshold=0.65,
        emb_sim_score=0.60,
        lambda_iou=0.05,  # 탐지-트랙 iou 신뢰도 결합 가중치
        lambda_mhd=0.05,  # 마하라노비스 거리 유사도 가중치
        lambda_shape=0.9,  # 형태 유사도 가중치
        use_dlo_boost=True,
        use_duo_boost=True,
        dlo_boost_coef=0.75,
        use_rich_s=True,
        use_sb=True,
        use_vt=True,
        s_sim_corr=True,
        use_reid=True,
        use_cmc=False,
        local_feature=True,
        feature_avg=True,
        model_type=args.model_type
    )
    return BoostTrack(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("BoostTrack for image sequence")
    parser.add_argument("--yolo_model", type=str, default="external/weights/yolo11x.pt")
    parser.add_argument("--img_path", type=str, default="data/cam2_labeld_add_rect")
    parser.add_argument("--model_type", type=str,
                       choices=['convNext', 'dinov2', 'swinv2',
                               'La_Transformer', 'VIT', 'DETR'],
                       default='dinov2')
    parser.add_argument("--reid_model", type=str, default=None)
    parser.add_argument('--emb_method', type=str, default='default',
                       choices=['default', 'mean', 'enhanced_mean', 'enhanced_mean_V2'])
    parser.add_argument('--visualize', action='store_true', default = True)
    parser.add_argument('--save_video', action='store_true', default = False)
    parser.add_argument('--save_frame', action='store_true', default = False)
    
    args = parser.parse_args()
    if args.reid_model is None:
        args.reid_model = ModelConfig.WEIGHTS[args.model_type]
        print(f"Using weights: {args.reid_model}")
    
    return args

def main():
    args = parse_args()
      
    img_files = natsorted([f for f in os.listdir(args.img_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    xml_files = natsorted([f for f in os.listdir(args.img_path) if f.endswith('.xml')])
    
    GeneralSettings.values['use_embedding'] = True
    GeneralSettings.values['use_ecc'] = False
    GeneralSettings.values['embedding_method'] = args.emb_method

    # YOLO 모델 초기화
    model = YOLO(args.yolo_model)
    
    # 트래커 모델 설정
    tracker = setup_tracker(args)
    
    # 시각화 모델 설정
    visualizer = Visualizer()

    # 평가지표 결과 저장
    all_metrics = [] # 모든 프레임의 MOTA, MOTP, IDF1 결과 저장
    
    # 시각화 설정
    vis_config = VisualizationConfig(
        visualize=args.visualize,
        save_video=args.save_video,
        save_frame=args.save_frame,
        stop_frame_ids=[i for i in range(len(img_files)) if i%100 == 0]
    )
    
    cam_name = args.img_path.split('/')[-1]

    save_dir = f'{args.model_type}_{cam_name}_view'
    model_type = ModelConfig.get_model_name(args.reid_model)
    
    video_writer = VideoWriter(save_dir, model_type, args.emb_method, os.path.basename(args.img_path)) if args.save_video else None
    
    valid_pairs = []
    for img_file in img_files:
        base_name = os.path.splitext(img_file)[0]
        xml_file = f"{base_name}.xml"
        not_found_list = []
        if xml_file in xml_files:
            valid_pairs.append((img_file, xml_file))
        else:
            not_found_list.append(img_file)
            

    # 이전 프레임 이미지 저장 덱
    previous_track_images = deque(maxlen=5)
    
    for idx, (img_name, xml_name) in enumerate(tqdm(valid_pairs)):
        frame_id = int(os.path.splitext(img_name)[0]) 
        img_path = os.path.join(args.img_path, img_name)
        xml_path = os.path.join(args.img_path, xml_name)
        
        np_img = cv2.imread(img_path)
        if np_img is None:
            continue
        
        # YOLO 모델 예측
        results = model.predict(np_img, device='cuda', classes=[0], iou=0.65, conf=0.5, verbose=False)
        
        # YOLO 탐지 결과 처리
        dets = process_yolo_detection(results) 
        
        # 필터링된 결과로 시각화
        yolo_plot = results[0].plot()
        if dets is not None:
            for i, box in enumerate(dets):
                x1, y1, x2, y2 = map(int, box[:4])
                # 박스 그리기
                cv2.rectangle(yolo_plot, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 인덱스 표시
                cv2.putText(yolo_plot, f"#{i}", (x1, y1-10),
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        
        # 프레임 정보 표시
        cv2.putText(yolo_plot, f"Frame: {idx} ({img_name})", (10, 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        
        # 검출객체가없다면 Image Pass
        if dets is None or len(dets) == 0: 
            continue
        
        # 이미지 전처리
        img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().cuda()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        # 객체 트래킹 업데이트
        targets = tracker.update(dets, img_tensor, np_img, str(frame_id)) # 반환 형태: [[x1, y1, x2, y2, id, score], ...]
        # 트래커 필터링
        tlwhs, ids, confs = utils.filter_targets(targets,
                                               GeneralSettings['aspect_ratio_thresh'], # 박스의 너비 / 높이 비율이 최대허용값 이상이면 필터링
                                               GeneralSettings['min_box_area']) # 박스의 넓이(픽셀단위)가 최소허용값 이하이면 필터링
        
        frame_mapping = {}
        track_boxes = []
        
        """
        해당코드는   Deprecation  예정
        """
        if dets is not None and ids is not None: # 무언가 탐지를한다면?
            for det_idx, yolo_box in enumerate(dets):
                max_iou = 0
                matched_id = None
                for tlwh, track_id in zip(tlwhs, ids):
                    x1, y1 = tlwh[0], tlwh[1]
                    x2, y2 = x1 + tlwh[2], y1 + tlwh[3]
                    track_box = [x1, y1, x2, y2]
                    track_boxes.append(track_box)
                    iou = utils.calculate_iou(yolo_box[:4], track_box) # yolo box 와 track box IOU를 통햬 먜칭진행
                    if iou > max_iou and iou > 0.6: # max이면서 겹칩정도가 적어도 N 이상은 되야함 
                        max_iou = iou
                        matched_id = track_id
                        
                if matched_id is not None:
                    frame_mapping[det_idx] = matched_id 

            yolo_tracking_id, box_mapping = visualizer.draw_detection_boxes(np_img, dets, frame_mapping)    #{yolo idx : [track_id , x1,y1,x2,y2,conf]}
        track_img, track_id_list = visualizer.draw_tracking_results(np_img, tlwhs, ids)
      
        # xml 레이블값들의 정보를 가져오고 시각화하는 함수임 레이블링 확인 코드
        xml_result = utils.get_bboxes_from_xml(xml_path) #  [xml_id , x1,y1,x2,y2 ] 
        xml_vis = visualizer.draw_xml_boxes(np_img, xml_result , idx) 

        '''''''''''''''''''''''''''''''''''''''''''''
        # MOT 계산
        '''''''''''''''''''''''''''''''''''''''''''''
        
        # YOLO 객체 탐지와 XML 박스를 매핑
        matched_results = utils.match_tracker_to_gt(targets, xml_path)
        frame_metrics = compute_mot_metrics(targets, xml_result, frame_id, matched_results)
        all_metrics.append(frame_metrics)


        cv2.putText(track_img, f"Frame: {frame_id}", (10, 30),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        
        # YOLO와 XML 매핑 시각화
        mapping_vis = visualizer.visualize_tracker_xml_mapping(np_img.copy(), targets, xml_result, matched_results)
        
        # Display results
        if vis_config.visualize :
            cv2.namedWindow('yolo_plot', cv2.WINDOW_NORMAL)
            cv2.imshow('yolo_plot', yolo_plot)
            cv2.namedWindow('yolo_tracking_id', cv2.WINDOW_NORMAL)
            cv2.imshow('yolo_tracking_id', yolo_tracking_id)
            cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
            cv2.imshow('Tracking', track_img)
            
            cv2.namedWindow('XML_GT', cv2.WINDOW_NORMAL)
            cv2.imshow('XML_GT', xml_vis)
            
            cv2.namedWindow('YOLO-XML Mapping', cv2.WINDOW_NORMAL)
            cv2.imshow('YOLO-XML Mapping', mapping_vis)
            
            # 이전 tracking 이미지 표시
            prev_window_names = []  # 이전 tracking 이미지 창 이름 저장
            for i, prev_img_info in enumerate(previous_track_images):
                window_name = f"Previous Track (Frame {prev_img_info['frame_id']}, XML ID: {prev_img_info['xml_id']}, Track ID: {prev_img_info['track_id']})"
                prev_window_names.append(window_name)
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, prev_img_info['image'])
            
            
            
            # 이전 tracking 이미지 창들만 닫기
            for window_name in prev_window_names:
                cv2.destroyWindow(window_name)

            
        # Write video frame
        if video_writer is not None:
            video_writer.write(track_img)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        
    
    final_mota, final_hota, final_idf1, total_fp, total_fn, total_idsw = get_final_mot_metrics(all_metrics)

    print(f"Final MOTA: {final_mota:.4f}")
    print(f"Final HOTA: {final_hota:.4f}")
    print(f"Final IDF1: {final_idf1:.4f}")
    print(f"Total False Positives: {total_fp}")
    print(f"Total False Negatives: {total_fn}")
    print(f"Total ID Switches: {total_idsw}")
    

    # 최종 결과 저장
    save_metrics(args.model_type, final_mota, final_hota, final_idf1, total_fp, total_fn, total_idsw)

    # Cleanup
    if video_writer is not None:
        video_writer.release()

if __name__ == "__main__":
    main()