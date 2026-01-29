# methods/base_solver.py
import os
import logging
import csv
import time
import datetime
import torch
import torch.nn as nn
from utils.data_utils import load_data, validate_model

# [수정] 모델 구조를 가져오기 위한 import
from quant_lib.qdrop.model.resnet import resnet18, resnet50
from quant_lib.qdrop.model.mobilenetv2 import mobilenetv2
# 필요한 다른 모델들도 여기 import 추가 가능 (regnet 등)

class BaseSolver:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. 실험 디렉토리 및 로거 설정
        self.setup_experiment_dir()
        
        # 2. 데이터 및 모델 로드
        self.train_loader, self.val_loader = self.build_data()
        self.model = self.build_model()

    def setup_experiment_dir(self):
        """
        logs/method/wXaX/timestamp 구조로 폴더 생성 및 로거 설정
        """
        bit_str = f"w{self.args.w_bit}a{self.args.a_bit}" if self.args.w_bit else "fp32"
        self.exp_root = os.path.join('logs', self.args.method, bit_str)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(self.exp_root, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = logging.getLogger(self.args.method)
        self.logger.setLevel(logging.INFO)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        
        # File Handler
        file_handler = logging.FileHandler(os.path.join(self.log_dir, 'log.txt'))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Stream Handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        
        self.logger.info(f"Experiment Directory Created: {self.log_dir}")
        self.logger.info(f"Args: {self.args}")

    def save_csv_result(self, top1, top5):
        csv_path = os.path.join(self.exp_root, 'summary.csv')
        file_exists = os.path.isfile(csv_path)
        
        result_data = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': self.args.model,
            'w_bit': self.args.w_bit,
            'a_bit': self.args.a_bit,
            'top1': round(top1, 4),
            'top5': round(top5, 4),
            'calib_iter': self.args.calib_iter,
            'log_dir': self.log_dir 
        }
        
        fieldnames = list(result_data.keys())
        
        with open(csv_path, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_data)
            
        self.logger.info(f"Results saved to {csv_path}")

    def build_data(self):
        return load_data(**self.config.data)

    def build_model(self):
        # 1. Config에 path가 명시되어 있으면 파일에서 로드
        if hasattr(self.config.model, 'path') and self.config.model.path is not None:
            if os.path.exists(self.config.model.path):
                self.logger.info(f"Loading model from path: {self.config.model.path}")
                from quant_lib.qdrop.model import load_model
                return load_model(self.config.model)
            else:
                self.logger.warning(f"[!] Warning: Model path '{self.config.model.path}' not found.")
                self.logger.info("    -> Falling back to torchvision pretrained weights.")

        # 2. Path가 없으면 라이브러리 내부 Custom Model 구조 + TV Pretrained Weight 사용
        try:
            import torchvision.models as tv_models
            model_name = self.config.model.type.lower()
            
            # (A) ResNet 계열
            if 'resnet' in model_name:
                # 1. Custom 구조 생성 (인자 제거!)
                if model_name == 'resnet18':
                    # [수정] pretrained=False 제거
                    model = resnet18() 
                elif model_name == 'resnet50':
                    # [수정] pretrained=False 제거
                    model = resnet50()
                else:
                    raise ValueError(f"Unsupported ResNet type for custom loading: {model_name}")
                
                # 2. Torchvision 가중치 로드
                self.logger.info(f"Loading torchvision pretrained weights for {model_name}...")
                tv_model = getattr(tv_models, model_name)(pretrained=True)
                
                # 3. 가중치 이식
                missing, unexpected = model.load_state_dict(tv_model.state_dict(), strict=False)
                if len(missing) > 0:
                    self.logger.warning(f"Missing keys: {missing}")
                
            # (B) MobileNetV2
            elif 'mobilenet' in model_name:
                # [수정] pretrained=False 제거
                model = mobilenetv2()
                self.logger.info(f"Loading torchvision pretrained weights for {model_name}...")
                tv_model = tv_models.mobilenet_v2(pretrained=True)
                model.load_state_dict(tv_model.state_dict(), strict=False)
                
            # (C) RegNet 등
            elif 'regnet' in model_name:
                self.logger.warning("RegNet custom loading not implemented yet. Using torchvision model directly.")
                model = getattr(tv_models, model_name)(pretrained=True)
                
            else:
                raise ValueError(f"Unknown model type: {model_name}")
                
            return model

        except Exception as e:
            self.logger.error(f"Failed to build model: {e}")
            raise e

    def validate(self):
        self.logger.info("Starting Validation...")
        # Validate 함수 내 device 처리는 validate_model 함수에 위임하거나 명시
        if next(self.model.parameters()).device.type == 'cpu':
             self.model.cuda()
             
        top1, top5 = validate_model(self.val_loader, self.model)
        self.logger.info(f"Validation Results - Top1: {top1:.2f}%, Top5: {top5:.2f}%")
        self.save_csv_result(top1, top5)
        
    def run(self):
        raise NotImplementedError