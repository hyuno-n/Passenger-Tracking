import warnings
from copy import deepcopy
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


def shape_similarity(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    from default_settings import BoostTrackSettings
    if not BoostTrackSettings['s_sim_corr']:
        return shape_similarity_v1(detects, tracks)
    else:
        return shape_similarity_v2(detects, tracks)

def shape_similarity_v1(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    if detects.size == 0 or tracks.size == 0:
        return np.zeros((0, 0))

    dw = (detects[:, 2] - detects[:, 0]).reshape((-1, 1))
    dh = (detects[:, 3] - detects[:, 1]).reshape((-1, 1))
    tw = (tracks[:, 2] - tracks[:, 0]).reshape((1, -1))
    th = (tracks[:, 3] - tracks[:, 1]).reshape((1, -1))
    return np.exp(-(np.abs(dw - tw)/np.maximum(dw, tw) + np.abs(dh - th)/np.maximum(dw, tw)))


def shape_similarity_v2(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    if detects.size == 0 or tracks.size == 0:
        return np.zeros((0, 0))

    dw = (detects[:, 2] - detects[:, 0]).reshape((-1, 1))
    dh = (detects[:, 3] - detects[:, 1]).reshape((-1, 1))
    tw = (tracks[:, 2] - tracks[:, 0]).reshape((1, -1))
    th = (tracks[:, 3] - tracks[:, 1]).reshape((1, -1))
    return np.exp(-(np.abs(dw - tw)/np.maximum(dw, tw) + np.abs(dh - th)/np.maximum(dh, th)))


def MhDist_similarity(mahalanobis_distance: np.ndarray, softmax_temp: float = 1.0) -> np.ndarray:
    limit = 13.2767  # 99% conf interval https://www.mathworks.com/help/stats/chi2inv.html
    mahalanobis_distance = deepcopy(mahalanobis_distance)
    mask = mahalanobis_distance > limit
    mahalanobis_distance[mask] = limit
    mahalanobis_distance = limit - mahalanobis_distance

    mahalanobis_distance = np.exp(mahalanobis_distance/softmax_temp) / np.exp(mahalanobis_distance/softmax_temp).sum(0).reshape((1, -1))
    mahalanobis_distance = np.where(mask, 0, mahalanobis_distance)
    return mahalanobis_distance


def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )

    return o


def soft_biou_batch(bboxes1, bboxes2):
    """
    Computes soft BIoU between two bboxes in the form [x1,y1,x2,y2]
    BIoU is introduced in https://arxiv.org/pdf/2211.14317
    Soft BIoU is introduced as part of BoostTrack++
    # Author : Vukasin Stanojevic
    # Email  : vukasin.stanojevic@pmf.edu.rs
    """

    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    k1 = 0.25
    k2 = 0.5
    b2conf = bboxes2[..., 4]
    b1x1 = bboxes1[..., 0] - (bboxes1[..., 2]-bboxes1[..., 0]) * (1-b2conf)*k1
    b2x1 = bboxes2[..., 0] - (bboxes2[..., 2]-bboxes2[..., 0]) * (1-b2conf)*k2
    xx1 = np.maximum(b1x1, b2x1)

    b1y1 = bboxes1[..., 1] - (bboxes1[..., 3]-bboxes1[..., 1]) * (1-b2conf)*k1
    b2y1 = bboxes2[..., 1] - (bboxes2[..., 3]-bboxes2[..., 1]) * (1-b2conf)*k2
    yy1 = np.maximum(b1y1, b2y1)

    b1x2 = bboxes1[..., 2] + (bboxes1[..., 2]-bboxes1[..., 0]) * (1-b2conf)*k1
    b2x2 = bboxes2[..., 2] + (bboxes2[..., 2]-bboxes2[..., 0]) * (1-b2conf)*k2
    xx2 = np.minimum(b1x2, b2x2)

    b1y2 = bboxes1[..., 3] + (bboxes1[..., 3]-bboxes1[..., 1]) * (1-b2conf)*k1
    b2y2 = bboxes2[..., 3] + (bboxes2[..., 3]-bboxes2[..., 1]) * (1-b2conf)*k2
    yy2 = np.minimum(b1y2, b2y2)

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h

    o = wh / (
        (b1x2 - b1x1) * (b1y2 - b1y1)
        + (b2x2 - b2x1) * (b2y2 - b2y1)
        - wh
    )

    return o


def match(cost_matrix: np.ndarray, threshold: float) -> np.ndarray:
    if cost_matrix.size > 0:
        a = (cost_matrix > threshold).astype(np.int32)
        # 탐지-트랙 매칭 결과가 1개인 경우
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        # 중복된 매칭 후보가 있는 경우
        else:
            cost_matrix = -cost_matrix  # maximize로 변경
            row_ind, col_ind = linear_sum_assignment(cost_matrix) # 헝가리 알고리즘 적용
            matched_indices = np.array([[row_ind[i], col_ind[i]] for i in range(len(row_ind))])
    else:
        matched_indices = np.empty(shape=(0, 2))
    return matched_indices


def linear_assignment(detections: np.ndarray, trackers: np.ndarray,
                      iou_matrix: np.ndarray, cost_matrix: np.ndarray,
                      threshold: float, emb_cost: Optional[np.ndarray] = None, emb_sim_score: float = 0.75):
    
    if iou_matrix is None and cost_matrix is None:
        raise Exception("Both iou_matrix and cost_matrix are None!")
    if iou_matrix is None:
        iou_matrix = deepcopy(cost_matrix)
    if cost_matrix is None:
        cost_matrix = deepcopy(iou_matrix)
        
    matched_indices = match(cost_matrix, threshold)
    
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        valid_match = emb_cost[m[0], m[1]] > emb_sim_score
        if valid_match:
            matches.append(m.reshape(1, 2))
        else:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers), cost_matrix


def associate(
        detections: np.ndarray,
        trackers: np.ndarray,
        iou_threshold: float,
        mahalanobis_distance: np.ndarray = None,
        track_confidence: np.ndarray = None,
        detection_confidence: np.ndarray = None,
        emb_cost: np.ndarray = None,
        emb_sim_score: float = None,
        lambda_iou: float = None,
        lambda_mhd: float = None,
        lambda_shape: float = None
):

    if emb_sim_score is None or lambda_iou is None or lambda_mhd is None or lambda_shape is None:
        print("emb_sim_score : {} lambda_iou : {} lambda_mhd : {} lambda_shape : {}".format(emb_sim_score, lambda_iou, lambda_mhd, lambda_shape))
        raise Exception("emb_sim_score, lambda_iou, lambda_mhd, lambda_shape must be specified!")
    
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
            np.empty((0, 0))
        )
    
    # 1. IoU 행렬 계산
    iou_matrix = iou_batch(detections, trackers)
    cost_matrix = deepcopy(iou_matrix)

    # 2. 탐지-트랙 신뢰도 결합 (Detection-Tracklet Confidence)
    if detection_confidence is not None and track_confidence is not None:
        conf = np.multiply(detection_confidence.reshape((-1, 1)), track_confidence.reshape((1, -1)))
        conf[iou_matrix < iou_threshold] = 0
        iou_for_cost = iou_batch(detections, trackers)
        cost_matrix += lambda_iou * conf * iou_for_cost
    else:
        warnings.warn("Detections or tracklet confidence is None and detection-tracklet confidence cannot be computed!")
        conf = None

    # 3. Mahalanobis 거리 기반 유사도 계산
    if mahalanobis_distance is not None and mahalanobis_distance.size > 0:
        mahalanobis_distance = MhDist_similarity(mahalanobis_distance)
        cost_matrix += lambda_mhd * mahalanobis_distance
        # 4. Shape 유사도 계산 (신뢰도 반영)
        if conf is not None:
            s = shape_similarity(detections, trackers)
            cost_matrix += lambda_shape * conf * s

    # 5. 임베딩 유사도 계산
    if emb_cost is not None:
        lambda_emb = (1 + lambda_iou + lambda_shape + lambda_mhd) * 1.5
        cost_matrix += lambda_emb * emb_cost

    # 최종 매칭 수행
    return linear_assignment(detections, trackers, iou_matrix, cost_matrix, iou_threshold, emb_cost, emb_sim_score)
