# qdrop_solver.py
import torch
import torch.nn as nn
import copy
import time
import logging

from .base_solver import BaseSolver
from quant_lib.qdrop.model import specials
from quant_lib.qdrop.quantization.state import enable_calibration_woquantization, enable_quantization, disable_all
from quant_lib.qdrop.quantization.quantized_module import QuantizedLayer, QuantizedBlock
from quant_lib.qdrop.quantization.fake_quant import QuantizeBase
from quant_lib.qdrop.quantization.observer import ObserverBase
from quant_lib.qdrop.solver.fold_bn import search_fold_and_remove_bn, StraightThrough
from quant_lib.qdrop.solver.recon import reconstruction

class QDropSolver(BaseSolver):
    def __init__(self, config, args):
        super().__init__(config, args)

    def run(self):
        """
        QDrop 실행 파이프라인
        """
        # 1. BN Folding
        self.logger.info("Step 1: Search and fold BN...")
        search_fold_and_remove_bn(self.model)
        
        # 2. 모델 구조 양자화 (QuantizedLayer/Block 래핑)
        self.logger.info(f"Step 2: Quantize Model Structure (W: {self.args.w_bit}bit, A: {self.args.a_bit}bit)")
        if hasattr(self.config, 'quant'):
            self.model = self._quantize_model_structure(self.model)
        
        self.model.cuda()
        self.model.eval()
        
        # Reconstruction용 FP32 모델 사본
        self.fp_model = copy.deepcopy(self.model)
        disable_all(self.fp_model)
        self.fp_model.eval()

        # Observer 이름 설정
        for name, module in self.model.named_modules():
            if isinstance(module, ObserverBase):
                module.set_name(name)

        # 3. Calibration (데이터 로드 및 Range 측정)
        # 중요: _get_cali_data 내부에서 전체 데이터를 가져옴
        cali_data = self._get_cali_data()
        
        self.logger.info(f"Step 3: Calibration ({len(cali_data)} samples)")
        self._calibrate(cali_data)

        # 4. Reconstruction
        if hasattr(self.config.quant, 'recon'):
            self.logger.info("Step 4: Start Reconstruction...")
            enable_quantization(self.model)
            
            # 재귀적으로 Reconstruction 수행
            self._reconstruct_recursive(self.model, self.fp_model, cali_data)
        
        # 5. 최종 평가
        self.logger.info("Step 5: Final Validation...")
        enable_quantization(self.model)
        self.validate()

    def _quantize_model_structure(self, model):
        """
        QDrop 방식의 모델 구조 변환
        """
        config_quant = self.config.quant

        def replace_module(module, w_qconfig, a_qconfig, qoutput=True):
            childs = list(iter(module.named_children()))
            st, ed = 0, len(childs)
            prev_quantmodule = None
            while(st < ed):
                tmp_qoutput = qoutput if st == ed - 1 else True
                name, child_module = childs[st][0], childs[st][1]
                
                # [핵심] BaseSolver 수정으로 인해 이제 child_module 타입이 specials 키와 일치하게 됨
                if type(child_module) in specials:
                    # Block 단위 래핑
                    setattr(module, name, specials[type(child_module)](child_module, w_qconfig, a_qconfig, tmp_qoutput))
                elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                    # Layer 단위 래핑 (첫/끝 레이어 등)
                    setattr(module, name, QuantizedLayer(child_module, None, w_qconfig, a_qconfig, qoutput=tmp_qoutput))
                    prev_quantmodule = getattr(module, name)
                elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                    if prev_quantmodule is not None:
                        prev_quantmodule.activation = child_module
                        setattr(module, name, StraightThrough())
                    else:
                        pass
                elif isinstance(child_module, StraightThrough):
                    pass
                else:
                    # 재귀 탐색
                    replace_module(child_module, w_qconfig, a_qconfig, tmp_qoutput)
                st += 1
        
        replace_module(model, config_quant.w_qconfig, config_quant.a_qconfig, qoutput=False)
        
        # Verify block-level wrapping
        self.logger.info("Verifying quantization structure:")
        block_count = 0
        layer_count = 0
        for name, module in model.named_modules():
            if isinstance(module, QuantizedBlock):
                block_count += 1
                if block_count <= 5:  # Log first 5 blocks
                    self.logger.info(f"  ✓ {name}: {type(module).__name__}")
            elif isinstance(module, QuantizedLayer):
                layer_count += 1
        
        self.logger.info(f"Quantization summary: {block_count} blocks, {layer_count} layers")
        if block_count == 0:
            self.logger.warning("WARNING: No QuantizedBlocks found! This may cause suboptimal results.")
        
        # 첫/마지막 레이어 8bit 고정
        w_list, a_list = [], []
        for name, module in model.named_modules():
            if isinstance(module, QuantizeBase) and 'weight' in name:
                w_list.append(module)
            if isinstance(module, QuantizeBase) and 'act' in name:
                a_list.append(module)
        
        if w_list: 
            w_list[0].set_bit(8)
            w_list[-1].set_bit(8)
        if a_list:
            #a_list[0].set_bit(8)
            a_list[-1].set_bit(8)
            #a_list[-1].disable_observer()   # Min/Max 측정 끄기
            #a_list[-1].disable_fake_quant() # Fake Quantization 연산 자체를 끄기 (Pass-through)
        
        self.logger.info("Model structure quantization complete.")
        return model

    def _get_cali_data(self):
        cali_data = []
        num_samples = self.config.quant.calibrate
        
        for batch in self.train_loader:
            cali_data.append(batch[0])
            if sum([x.size(0) for x in cali_data]) >= num_samples:
                break
        
        cali_data = torch.cat(cali_data, dim=0)[:num_samples]
        self.logger.info(f"Calibration data loaded: {cali_data.shape}, device: {cali_data.device}")
        
        if torch.cuda.is_available():
            cali_data = cali_data.cuda()
            self.logger.info(f"Calibration data moved to CUDA: {cali_data.device}")
            
        return cali_data

    def _calibrate(self, cali_data):
        self.logger.info(f"Starting calibration with data shape: {cali_data.shape}, device: {cali_data.device}")
        
        with torch.no_grad():
            st = time.time()
            # 1. Activation Calibration (256 samples as per official QDrop)
            enable_calibration_woquantization(self.model, quantizer_type='act_fake_quant')
            act_samples = min(256, cali_data.shape[0])
            self.logger.info(f"Running activation calibration with {act_samples} samples")
            
            # Process in smaller batches to avoid OOM
            batch_size = 32
            for i in range(0, act_samples, batch_size):
                batch_end = min(i + batch_size, act_samples)
                self.model(cali_data[i:batch_end])
                if (i + batch_size) % 128 == 0:
                    torch.cuda.empty_cache()  # Clear cache periodically
            
            # 2. Weight Calibration (2 samples as per official QDrop)
            enable_calibration_woquantization(self.model, quantizer_type='weight_fake_quant')
            weight_samples = min(2, cali_data.shape[0])
            self.logger.info(f"Running weight calibration with {weight_samples} samples")
            self.model(cali_data[:weight_samples])
            
            torch.cuda.empty_cache()  # Clear cache after calibration
            ed = time.time()
            self.logger.info(f"Calibration time: {ed - st:.2f}s")

    def _reconstruct_recursive(self, module: nn.Module, fp_module: nn.Module, cali_data: torch.Tensor, prefix=""):
        for name, child_module in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child_module, (QuantizedLayer, QuantizedBlock)):
                # Block 또는 Layer 단위 Reconstruction
                module_type = "Block" if isinstance(child_module, QuantizedBlock) else "Layer"
                self.logger.info(f"Reconstruction for {module_type}: {full_name} (type: {type(child_module).__name__})")
                
                reconstruction(
                    self.model, 
                    self.fp_model, 
                    child_module, 
                    getattr(fp_module, name), 
                    cali_data, 
                    self.config.quant.recon
                )
            else:
                self._reconstruct_recursive(child_module, getattr(fp_module, name), cali_data, full_name)