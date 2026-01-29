import torch
import torch.nn as nn
from .base_solver import BaseSolver

# BRECQ Library Import
from quant import quant_model
from quant import block_recon
from quant import fold_bn

# [Magic Patch] nn.Module에 set_quant_state 기능 주입 (안전장치)
def set_quant_state_wrapper(self, weight_quant: bool = False, act_quant: bool = False):
    for m in self.modules():
        if isinstance(m, quant_model.QuantModule):
            m.set_quant_state(weight_quant, act_quant)

if not hasattr(torch.nn.Module, 'set_quant_state'):
    torch.nn.Module.set_quant_state = set_quant_state_wrapper

class BRECQSolver(BaseSolver):
    def run(self):
        # Debugging Tool
        from utils.data_utils import validate_model
        
        # 1. Config Loading
        # -------------------------------------------------------
        w_bit = self.config.quant.w_qconfig.bit
        a_bit = self.config.quant.a_qconfig.bit
        
        # W2, Rounding Weight Recommanded 0.1
        rounding_weight = self.config.brecq.weight
        if w_bit == 2 and rounding_weight < 0.1:
            self.logger.warning(f"[!] Warning: W2 experiment detected but weight is {rounding_weight}. Recommended: 0.1")
        
        self.logger.info(f"Step 1: Initialize QuantModel (W:{w_bit} / A:{a_bit})")
        
        # 2. Model Initialization
        # QuantModule은 기본적으로 UniformAffineQuantizer로 초기화됩니다.
        # (AdaRound는 나중에 block_recon에서 동적으로 교체되므로 여기서 설정할 필요 없음)
        self.q_model = quant_model.QuantModel(
            model=self.model, 
            weight_quant_params={'n_bits': w_bit, 
                                 'channel_wise': True,
                                 'scale_method': 'mse'
                                 }, 
            act_quant_params={'n_bits': a_bit,
                              'leaf_param': True,
                              'scale_method': 'mse'
                              }
        )
        self.q_model.cuda()
        self.q_model.eval()
        
        # 3. First & Last Layer 8-bit Protection (Crucial for W2)
        self.logger.info("Step 2: Setting first/last layer to 8-bit...")
        self.q_model.set_first_last_layer_to_8bit()
        
        # [CHECK 1] FP32 Baseline Verification
        self.logger.info("[CHECK 1] Checking QuantModel Wrapper (Quant OFF)...")
        self.q_model.set_quant_state(weight_quant=False, act_quant=False)
        acc1, acc5 = validate_model(self.val_loader, self.q_model)
        self.logger.info(f" -> Result: Top1 {acc1:.2f}% (Should match official weights)")

        # 4. Calibration
        self.logger.info("Step 3: Calibration...")
        
        # Activation Range Calibration
        self._calibrate_activation()
        
        # Initialize weight quantizer deltas (Uniform Initialization)
        self.logger.info("Initializing weight quantization parameters...")
        self.q_model.set_quant_state(weight_quant=True, act_quant=False)
        
        # Use 64 samples for initialization (Consistent with paper code)
        with torch.no_grad():
            _ = self.q_model(self._get_cali_data()[:64])
            
        # [CHECK 2] Accuracy Before Reconstruction
        self.logger.info("[CHECK 2] Checking QuantModel after Calibration (Quant ON)...")
        self.q_model.set_quant_state(weight_quant=True, act_quant=True)
        acc1, acc5 = validate_model(self.val_loader, self.q_model)
        self.logger.info(f" -> Result: Top1 {acc1:.2f}% (Before reconstruction)")


        # 5. Reconstruction Phase 1: Weight Reconstruction (AdaRound)
        # -------------------------------------------------------
        self.logger.info("Step 4: Weight Reconstruction (Phase 1)...")
        self.logger.info(f" -> Using AdaRound for W{w_bit} optimization (Dynamic Swap in block_recon)")
        
        cali_data = self._get_cali_data()
        
        # Extract Module List
        module_list = []
        # ResNet Layers
        resnet_layers = ['layer1', 'layer2', 'layer3', 'layer4']
        for name in resnet_layers:
            if hasattr(self.q_model.model, name):
                layer_container = getattr(self.q_model.model, name)
                for block in layer_container:
                    module_list.append(block)
        
        # MobileNetV2 Layers
        if not module_list and hasattr(self.q_model.model, 'features'):
             for block in self.q_model.model.features:
                 if type(block).__name__ == 'QuantInvertedResidual': 
                     module_list.append(block)
        
        # Params Setup
        iters_w = self.config.brecq.get('iters_w', self.config.brecq.iters)
        iters_a = self.config.brecq.get('iters_a', 5000)
        keep_gpu = self.config.brecq.get('keep_gpu', True)
        
        # Phase 1 Loop
        for i, block in enumerate(module_list):
            self.logger.info(f"[Weight] Reconstructing Block [{i+1}/{len(module_list)}]: {block.__class__.__name__}")

            # block recon에서 AdaRoundQuantizer로 알아서 교체됨
            block_recon.block_reconstruction(
                self.q_model, 
                block, 
                cali_data, 
                batch_size=self.config.brecq.batch_size, 
                iters=iters_w, 
                weight=self.config.brecq.weight, # 2비트일 때 0.1이어야 함
                opt_mode='mse',
                asym=True,
                b_range=self.config.brecq.b_range,
                warmup=self.config.brecq.warm_up,
                act_quant=False,                 # False면 Weight 최적화 (AdaRound)
                lr=self.config.brecq.lr,
                p=self.config.brecq.p,
                keep_gpu=keep_gpu
            )
        
        # Checkpoint: Weight-only Acc
        self.logger.info("[Checkpoint] Weight-only quantization accuracy:")
        self.q_model.set_quant_state(weight_quant=True, act_quant=False)
        acc1, acc5 = validate_model(self.val_loader, self.q_model)
        self.logger.info(f" -> Weight-only: Top1 {acc1:.2f}%")
        
        # 6. Reconstruction Phase 2: Activation Reconstruction
        # -------------------------------------------------------
        skip_phase2 = self.config.brecq.get('skip_phase2', False)
        
        if a_bit < 8 and not skip_phase2:
            self.logger.info("Step 5: Activation Reconstruction (Phase 2)...")
            
            # Re-init activation scale
            self.q_model.set_quant_state(weight_quant=True, act_quant=True)
            with torch.no_grad():
                _ = self.q_model(cali_data[:64])
            
            # Disable Output Quantization (Crucial Fix)
            self.logger.info("Disabling network output quantization for Phase 2...")
            if hasattr(self.q_model, 'disable_network_output_quantization'):
                self.q_model.disable_network_output_quantization()
            
            # Phase 2 Loop
            for i, block in enumerate(module_list):
                self.logger.info(f"[Activation] Reconstructing Block [{i+1}/{len(module_list)}]: {block.__class__.__name__}")

                block_recon.block_reconstruction(
                    self.q_model, 
                    block, 
                    cali_data, 
                    batch_size=self.config.brecq.batch_size, 
                    iters=iters_a,
                    weight=self.config.brecq.weight,
                    opt_mode='mse',
                    asym=False,
                    b_range=self.config.brecq.b_range,
                    warmup=self.config.brecq.warm_up,
                    act_quant=True,             # True면 Activation 최적화 (Learned Step Size)
                    lr=4e-4,                    # Phase 2 LR
                    p=self.config.brecq.p,
                    keep_gpu=keep_gpu
                )
        elif skip_phase2:
            self.logger.info("Step 5: Skipping Activation Reconstruction (Phase 2)")
        
        # 7. Final Validation
        self.logger.info("Step 6: Final Validation...")
        torch.cuda.empty_cache()
        self.q_model.set_quant_state(weight_quant=True, act_quant=True)
        self.validate_brecq()

    def _get_cali_data(self):
        cali_data = []
        num_samples = self.config.quant.calibrate
        for batch in self.train_loader:
            cali_data.append(batch[0])
            if len(cali_data) * batch[0].size(0) >= num_samples:
                break
        return torch.cat(cali_data, dim=0)[:num_samples].cuda()

    def _calibrate_activation(self):
        self.logger.info("Collecting activation statistics...")
        self.q_model.set_quant_state(weight_quant=False, act_quant=True)
        
        cali_data = self._get_cali_data()
        with torch.no_grad():
            batch_size = 32
            for i in range(0, len(cali_data), batch_size):
                self.q_model(cali_data[i : i + batch_size])
        
        self.logger.info("Activation statistics collected.")
                
    def validate_brecq(self):
        from utils.data_utils import validate_model
        top1, top5 = validate_model(self.val_loader, self.q_model)
        self.logger.info(f"Validation Results - Top1: {top1:.2f}%, Top5: {top5:.2f}%")
        self.save_csv_result(top1, top5)