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
            'VIT-B/16+ICS_SSL': VITSSLModelCreator(),
            'convNext': ConvNextModelCreator(),
        }
        
        creator = model_creators.get(config.model_name)
        if not creator:
            raise ValueError(f"Unsupported model type: {config.model_name}")
        
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
        model.load_state_dict(torch.load(config.reid_model_path), weights_only=True, strict=False)
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

class ModelConfigFactory:
    """Factory for creating model configurations"""
    @staticmethod
    def create_config(model_name: str) -> ModelConfig:
        configs = {
            'dinov2': ModelConfig(crop_size=(448, 448), model_type='dinov2'),
            'swinv2': ModelConfig(crop_size=(192, 192), model_type='swinv2'),
            'La_Transformer': ModelConfig(crop_size=(224, 224), model_type='La_Transformer'),
            'VIT-B/16+ICS_SSL': ModelConfig(crop_size=(256, 128), model_type='VIT-B/16+ICS_SSL'),
            'convNext': ModelConfig(crop_size=(384, 384), model_type='convNext'),
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
            'VIT-B/16+ICS_SSL': DefaultProcessor(),
        }
        return processors.get(model_type, DefaultProcessor())


class EmbeddingComputer:
    def __init__(self, config):
        self.model = None
        self.config = config
        self.model_type = config.model_name  # 올바른 속성 사용
        self.model_config = ModelConfigFactory.create_config(config.model_name)

        # 모델별 전처리 설정
        self.transform = self.get_transform(self.model_config.crop_size, config.model_name)

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
            "VIT-B/16+ICS_SSL": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            "convNext": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

            for j, box in enumerate(batch_bbox):
                if j < len(batch_embeddings):
                    x1, y1, x2, y2 = map(int, box)
                    box_key = f"{tag}_{x1}_{y1}_{x2}_{y2}"
                    self.cache[box_key] = batch_embeddings[j]
                    embeddings.append(batch_embeddings[j])

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