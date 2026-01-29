import copy
import torch
import torch.nn as nn
import logging

from quant_lib.adaround.quant import Quantizer, QuantizedLayer, QuantizedBlock
from quant_lib.adaround.quant.fake_quant import QuantizeBase 
from quant_lib.adaround.quant.recon import reconstruction

from .base_solver import BaseSolver
from utils.data_utils import validate_model 

logger = logging.getLogger('adaround_solver')

class AdaRoundSolver(BaseSolver):
    def __init__(self, config, args):
        super().__init__(config,  args)
        
        self.fp_model = copy.deepcopy(self.model)
        self.fp_model.cuda()
        self.fp_model.eval()
        
        logger.info("Wrapping model into QuantizedModule...")
        self.quant_model = Quantizer(self.model, self.config.quant.w_qconfig)
        self.quant_model.cuda()
        self.quant_model.eval()
        
    def get_calib_data(self):
        calib_data = []
        total_samples = 0
        
        for i, (data, target) in enumerate(self.train_loader):
            calib_data.append(data)
            total_samples += data.size(0)
            if total_samples >= self.config.quant.calibrate:
                break
        
        return torch.cat(calib_data, dim=0)[:self.config.quant.calibrate]
    
    def set_quant_state(self, mode='calibration'):
        """
        MODE: 
          - 'calibration': Observer ON, FakeQuant OFF
          - 'eval': Observer OFF, FakeQuant ON
        """
        for name, module in self.quant_model.named_modules():
            if isinstance(module, QuantizeBase):
                if mode == 'calibration':
                    module.enable_observer()
                    module.disable_fake_quant()
                elif mode == 'eval':
                    module.disable_observer()
                    module.enable_fake_quant()
                    
    def get_module_by_name(self, model, target_name):
        for name, module in model.named_modules():
            if name == target_name:
                return module
        return None
    
    def run(self):
        # Step 1: Calibration
        logger.info("Step 1: Calibrating Activation Statistics(MinMax)...")
        
        calib_data = self.get_calib_data()
        
        self.set_quant_state(mode='calibration')
        
        with torch.no_grad():
            bs = self.config.data.batch_size
            for i in range(0, calib_data.size(0), bs):
                batch = calib_data[i:i+bs].cuda()
                self.quant_model(batch)
                
        self.set_quant_state(mode='eval')
        logger.info("Activation calibration finished")
        
        # Step 2: AdaRound Reconstruction(Block-wise)
        logger.info("Step 2: Running AdaRound Reconstruction...")
        
        for name, q_module in self.quant_model.named_modules():
            if isinstance(q_module, (QuantizedLayer, QuantizedBlock)):
                logger.info(f"Processing Block: {name}...")
                
                fp_module = self.get_module_by_name(self.fp_model, name)
                if fp_module is None:
                    logger.warning(f"Could not find matching FP module for {name}, skipping.")
                    continue
            
                reconstruction(
                    model = self.quant_model,
                    fp_model = self.fp_model,
                    module = q_module,
                    fp_module = fp_module,
                    cali_data = calib_data,
                    config = self.config.quant.recon
                )
        
        # Step 3: Final Validation
        logger.info("Step 3: Validating Quantized Model after AdaRound...")
        
        top1, top5 = validate_model(self.val_loader, self.quant_model)
        
        logger.info(f"AdaRound Final Result - Top1: {top1:.2f}%, Top5: {top5:.2f}%")
        self.save_csv_result(top1, top5)