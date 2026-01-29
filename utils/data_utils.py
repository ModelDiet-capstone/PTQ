import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import logging
import os
from easydict import EasyDict

class DummyDataset(data.Dataset):
    """
    실제 이미지가 없을 때 테스트를 위해 랜덤 노이즈를 뱉어내는 가짜 데이터셋
    """
    def __init__(self, length=1000):
        self.length = length
        # ImageNet 크기 (3, 224, 224)
        self.data = torch.randn(length, 3, 224, 224)
        # 랜덤 라벨 (0~999)
        self.target = torch.randint(0, 1000, (length,))

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.length

def parse_config(config_file):
    import yaml
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # Dict를 EasyDict로 변환 (속성 접근 편의성)
    return EasyDict(config)

def load_data(path, batch_size, num_workers, pin_memory, input_size, test_resize, **kwargs):
    # 실제 데이터 경로가 있으면 로드, 없으면 더미 데이터 사용
    
    # [수정됨] 폴더 구조 유연하게 처리
    if os.path.exists(path):
        # 1. 표준 ImageNet 구조인지 확인 (train 폴더가 있는지)
        if os.path.exists(os.path.join(path, 'train')):
            traindir = os.path.join(path, 'train')
            valdir = os.path.join(path, 'val')
            print(f"[*] Found standard ImageNet structure at: {path}")
        else:
            # 2. train 폴더가 없으면 입력된 경로 자체를 데이터 루트로 사용
            # (Validation 데이터만 가지고 있을 때, 이를 Calibration용으로도 사용)
            traindir = path
            valdir = path
            print(f"[*] Standard 'train' folder not found. Using root path for both train/val: {path}")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        # Train Dataset (Calibration용 - Random Crop 등 Augmentation 적용)
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        
        # Validation Dataset (평가용 - Center Crop)
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(test_resize),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ]))
            
    else:
        # --- 더미 데이터 로드 로직 (경로가 아예 없을 때) ---
        print(f"Warning: Dataset path '{path}' not found. Using Dummy Data for testing!")
        train_dataset = DummyDataset(length=1280) # Calibration용
        val_dataset = DummyDataset(length=500)    # Validation용

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=None
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader

def validate_model(val_loader, model):
    """
    모델 평가 함수 (Top-1, Top-5 Acc)
    """
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.eval()
    
    # QDrop Hook 등에서 사용하는 예외 처리용 클래스 필요 시 여기에 정의
    # (원래 imagenet_utils에 있던 것들)
    
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()

            # compute output
            output = model(images)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            if i % 10 == 0:
                print(f'Test: [{i}/{len(val_loader)}]\tAcc@1 {top1.val:.3f} ({top1.avg:.3f})')

    return top1.avg, top5.avg

# --- Helper Classes ---
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# QDrop recon.py에서 import하는 클래스들 더미 정의 (에러 방지용)
class StopForwardException(Exception):
    pass

class DataSaverHook:
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward
        self.input_store = None
        self.output_store = None
    def __call__(self, module, input_batch, output_batch):
        if self.store_input: self.input_store = input_batch
        if self.store_output: self.output_store = output_batch
        if self.stop_forward: raise StopForwardException