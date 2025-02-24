from ast import Import
from collections import OrderedDict
from pathlib import Path
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import torchvision
import numpy as np
import torchvision.transforms as T
from .TransReID_SSL.transreid_pytorch.model.backbones.vit_pytorch import vit_base_patch16_224_TransReID as VIT_EXTEND
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt

def convert_dict_to_namespace(config_dict):
    """
    dict를 argparse.Namespace로 변환하여 build() 함수와 호환되도록 함.
    """
    namespace = argparse.Namespace()
    for key, value in config_dict.items():
        setattr(namespace, key, value)
    return namespace

@dataclass
class ModelConfig:
    """Base configuration class for all models"""
    crop_size: Tuple[int, int]
    model_type: str
    
    def get_transform(self) -> T.Compose:
        return T.Compose([
            T.ToPILImage(),
            T.Resize(self.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class ModelFactory:
    """Factory class for creating model instances"""
    @staticmethod
    def create_model(config, device):
        model_creators = {
            'dinov2': Dinov2ModelCreator(),
            'swinv2': SwinV2ModelCreator(),
            'La_Transformer': LATransformerModelCreator(),
            'VIT': VITSSLModelCreator(),
            'convNext': ConvNextModelCreator(),
            'DETR': DETRModelCreator(),
        }
        
        creator = model_creators.get(config.model_type)
        if not creator:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        return creator.create_model(config, device)

class BaseModelCreator:
    """Base class for model creators"""
    def create_model(self, config, device):
        raise NotImplementedError

class Dinov2ModelCreator(BaseModelCreator):
    def create_model(self, config, device):
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        model.to(device)
        model.eval()
        return model

class SwinV2ModelCreator(BaseModelCreator):
    def create_model(self, config, device):
        from .MS_Swin_Transformer.models.swin_transformer_v2 import SwinTransformerV2 as MS_Swin_Transformer_V2
        init_args = {
            'img_size': 192,
            'patch_size': 4,
            'embed_dim': 192,
            'depths': [2, 2, 18, 2],
            'num_heads': [6, 12, 24, 48],
            'window_size': 12,
            'mlp_ratio': 4.0,
            'qkv_bias': True,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.2,
            'patch_norm': True,
            'use_checkpoint': False,
            'pretrained_window_sizes': [12, 12, 12, 12]
        }
        model = MS_Swin_Transformer_V2(**init_args)
        model.to(device)
        if hasattr(model, 'head'):
            model.head = torch.nn.Identity().to(device)
        model.eval()
        return model

class LATransformerModelCreator(BaseModelCreator):
    def create_model(self, config, device):
        import timm
        from tracker.LA_Transformer.LATransformer.model import LATransformer
        base_model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=751
        )
        model = LATransformer(base_model, lmbd=0.2)
        model.load_state_dict(torch.load(config.reid_model_path), strict=False)
        model.to(device)
        model.eval()
        return model

class VITSSLModelCreator(BaseModelCreator):
    def create_model(self, config, device):
        model = VIT_EXTEND(
            img_size=(256, 128),
            stride_size=16,
            drop_path_rate=0.1,
            camera=0,
            view=0,
            sie_xishu=0.0,
            local_feature=False,
            gem_pool=True,
            stem_conv=True,
            num_classes=0
        )
        model.load_state_dict(torch.load(config.reid_model_path), strict=False)
        model.to(device)
        return model

class ConvNextModelCreator(BaseModelCreator):
    def create_model(self, config, device):
        from .ConvNeXt.models.convnext import ConvNeXt
        config_model ={
            'xlarge':{
                'depths':[3, 3, 27, 3],
                'dims':[256, 512, 1024, 2048]
            },
            'large':{
                'depths':[3, 3, 27, 3],
                'dims':[192, 384, 768, 1536]
            },
            'base':{
                'depths':[3, 3, 27, 3],
                'dims':[128, 256, 512, 1024]                            
            },
            'small':{
                'depths':[3, 3, 27, 3],
                'dims':[96, 192, 384, 768]
            }
            
        }
        SIZE = str(config.reid_model_path).split('/')[-1].split('_')[1]
        model = ConvNeXt(
            depths = config_model[SIZE]['depths'],
            dims = config_model[SIZE]['dims']
        )
        model.to(device)
        if hasattr(model, 'head'):
            model.head = torch.nn.Identity().to(device)
        model.eval()
        return model

class DETRModelCreator(BaseModelCreator):
    def create_model(self, config, device):
        from tracker.Deformable_DETR.models.deformable_detr import build
        # 🔍 Deformable DETR 모델 설정
        args = {
            "hidden_dim": 256,               # 모델의 임베딩 차원
            "nheads": 8,                     # Multi-head Attention 개수
            "enc_layers": 6,                 # Transformer Encoder 레이어 개수
            "dec_layers": 6,                 # Transformer Decoder 레이어 개수
            "dim_feedforward": 1024,         # Feedforward layer 차원
            "dropout": 0.1,                   # Dropout 비율
            "num_feature_levels": 4,          # Feature Pyramid Level 개수
            "num_queries": 300,               # Object Query 개수 (COCO의 경우 100 추천)
            "aux_loss": True,                 # Auxiliary loss 사용 여부
            "with_box_refine": False,         # Iterative bounding box refinement 사용 여부
            "two_stage": False,               # Two-stage DETR 사용 여부
            "dataset_file": "coco",           # 데이터셋 설정
            "device": "cuda" if torch.cuda.is_available() else "cpu",  # 실행 장치 설정
            "reid_model_path": "external/weights/r50_deformable_detr.pth",  # 모델 가중치 파일
            "position_embedding": "sine",     # Positional Embedding 설정
            "lr_backbone": 1e-5,              # Backbone의 학습률
            "masks": False,                   # Mask Head 사용 여부
            "backbone": "resnet50",           # Backbone 모델 설정
            "dilation": False,                # Backbone Dilated Conv 사용 여부
            "dec_n_points": 4,                # Deformable Attention의 sampling points 개수
            "enc_n_points": 4,                # Deformable Attention의 sampling points 개수
            "set_cost_class": 2,              # Set transformer의 class cost 가중치
            "set_cost_bbox": 5,               # Set transformer의 bbox cost 가중치
            "set_cost_giou": 2,               # Set transformer의 giou cost 가중치
            "bbox_loss_coef": 5,              # Bounding box loss 가중치
            "cls_loss_coef": 2,               # Class loss 가중치
            "giou_loss_coef": 2,              # GIoU loss 가중치
            "focal_alpha": 0.25,              # Focal loss의 alpha 값
        }

        detr_args = convert_dict_to_namespace(args)

        # 모델 인스턴스 생성
        model,_,_ = build(detr_args)
        checkpoint = torch.load(config.reid_model_path, map_location=device)
        if "model" in checkpoint:  # 일반적인 DETR 체크포인트 포맷
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model

class ModelConfigFactory:
    """Factory for creating model configurations"""
    @staticmethod
    def create_config(model_name: str) -> ModelConfig:
        configs = {
            'dinov2': ModelConfig(crop_size=(448, 448), model_type='dinov2'),
            'swinv2': ModelConfig(crop_size=(192, 192), model_type='swinv2'),
            'La_Transformer': ModelConfig(crop_size=(224, 224), model_type='La_Transformer'),
            'VIT': ModelConfig(crop_size=(256, 128), model_type='VIT'),
            'convNext': ModelConfig(crop_size=(384, 384), model_type='convNext'),
            'DETR': ModelConfig(crop_size=(800, 800), model_type='DETR'),
        }
        
        if model_name not in configs:
            raise ValueError(f"Unsupported model type: {model_name}")
        
        return configs[model_name]


class BatchEmbeddingProcessor:
    """배치 임베딩 처리를 위한 기본 클래스"""
    def process_batch(self, batch_input, model, batch_image=None):
        raise NotImplementedError

class DefaultProcessor(BatchEmbeddingProcessor):
    def process_batch(self, batch_input, model, batch_image=None):
        return model(batch_input)

class BatchProcessorFactory:
    """배치 프로세서 생성을 위한 팩토리 클래스"""
    @staticmethod
    def create_processor(model_type: str) -> BatchEmbeddingProcessor:
        processors = {
            'La_Transformer': DefaultProcessor(),
            'swinv2': DefaultProcessor(),
            'convNext': DefaultProcessor(),
            'VIT': DefaultProcessor(),
            'DETR': DefaultProcessor(),
        }
        return processors.get(model_type, DefaultProcessor())


class EmbeddingComputer:
    def __init__(self, config):
        self.model = None
        self.config = config
        self.model_type = config.model_type  # 올바른 속성 사용
        self.model_config = ModelConfigFactory.create_config(config.model_type)

        # 모델별 전처리 설정
        self.transform = self.get_transform(self.model_config.crop_size, config.model_type)

        print("Model Input size:", self.model_config.crop_size)

        self.max_batch = 8
        self.device = torch.device("cuda")
        self.initialize_model()

        self.cache = {}
        self.cache_name = ""

    def get_transform(self, crop_size, model_type):
        """
        모델별로 최적화된 전처리를 적용하는 함수
        """
        normalize_params = {
            "dinov2": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            "swinv2": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            "La_Transformer": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            "VIT": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            "convNext": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            "DETR": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        }

        mean, std = normalize_params.get(model_type, ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

        return T.Compose([
            T.ToPILImage(),
            T.Resize(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def compute_embedding(self, img, bbox, tag):
        """
        이미지에서 객체의 임베딩을 계산하는 함수
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        if len(bbox) == 0:
            return np.array([])

        if tag != self.cache_name:
            self.cache = {}
            self.cache_name = tag

        batch_size = min(self.max_batch, len(bbox))
        n_batches = math.ceil(len(bbox) / batch_size)
        batch_image = []
        embeddings = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(bbox))
            batch_bbox = bbox[start_idx:end_idx]

            batch_tensors = []
            for box in batch_bbox:
                x1, y1, x2, y2 = map(int, box)
                box_key = f"{tag}_{x1}_{y1}_{x2}_{y2}"

                if box_key in self.cache:
                    embedding = self.cache[box_key]
                else:
                    cropped = img[y1:y2, x1:x2]
                    batch_image.append(cropped)
                    if cropped.size == 0:
                        continue

                    tensor = self.transform(cropped)
                    batch_tensors.append(tensor)

            if not batch_tensors:
                continue

            batch_input = torch.stack(batch_tensors).to(self.device, non_blocking=True)
            processor = BatchProcessorFactory.create_processor(self.model_type)

            with torch.no_grad():
                batch_embeddings = processor.process_batch(batch_input, self.model, batch_image)
                batch_embeddings = self.process_output(batch_embeddings)
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                batch_embeddings = batch_embeddings.cpu().numpy()
            print(batch_bbox)
            for j, box in enumerate(batch_bbox):
                if j < len(batch_embeddings):
                    x1, y1, x2, y2 = map(int, box)
                    box_key = f"{tag}_{x1}_{y1}_{x2}_{y2}"
                    self.cache[box_key] = batch_embeddings[j]
                    embeddings.append(batch_embeddings[j])
                    # 박스를 시각화 (중복 실행 방지)
                    cropped = img[y1:y2, x1:x2]
                    cv2.namedWindow(f"Object {j+1}", cv2.WINDOW_NORMAL)
                    cv2.imshow(f"Object {j+1}", cropped)
            cv2.waitKey(0)  # 키 입력을 기다림
            cv2.destroyAllWindows()  # 열린 모든 창을 닫음
            print(batch_embeddings)
        return np.array(embeddings) if embeddings else np.array([])

    def process_output(self, output):
        """
        모델별로 다른 출력 형태를 통합하는 후처리 함수
        """
        if isinstance(output, dict):
            merged_output = torch.cat([v for v in output.values()], dim=-1)  # (batch_size, total_feature_dim)
            return merged_output
        elif isinstance(output, torch.Tensor):  # 일반적인 경우
            return output

        else:
            raise TypeError(f"Unexpected output type from model: {type(output)}")

    def initialize_model(self):
        self.model = ModelFactory.create_model(self.config, self.device)