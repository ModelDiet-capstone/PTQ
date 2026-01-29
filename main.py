import sys
import os
import argparse
import logging
import yaml
import torch
import torch.nn as nn
from easydict import EasyDict

# ---------------------------------------------------------
# [1] Basic Path Setup
# Always register the current file location (PTQ root)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 'imagenet_utils' Aliasing (Common)
# Link utils.data_utils when external libraries look for 'imagenet_utils'
from utils import data_utils
sys.modules['imagenet_utils'] = data_utils
# ---------------------------------------------------------

from utils.data_utils import parse_config 
from methods import get_solver_class

def get_args():
    parser = argparse.ArgumentParser(description='Quantization Experiment Platform')
    
    # Method Selection
    parser.add_argument('--method', type=str, default='qdrop', 
                        choices=['qdrop', 'brecq', 'adaround'], help='Method to run')
    
    # Config File Path (Auto-detected if None)
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    
    # Override Arguments (For overwriting experiment variables)
    parser.add_argument('--model', type=str, default=None, help='Model name override')
    parser.add_argument('--w_bit', type=int, default=None, help='Weight bit-width')
    parser.add_argument('--a_bit', type=int, default=None, help='Activation bit-width')
    parser.add_argument('--calib_iter', type=int, default=None, help='Calibration iterations')
    
    # Other Options
    parser.add_argument('--seed', type=int, default=1005, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='logs', help='Log directory')
    
    args = parser.parse_args()
    return args

def merge_config(config, args):
    """Overwrite Config values with CLI arguments."""
    if args.model:
        if 'type' in config.model: config.model.type = args.model
        elif 'name' in config.model: config.model.name = args.model
            
    # Overwrite only if Quant Config structure exists (Handling BRECQ/QDrop structure differences)
    if hasattr(config, 'quant'):
        if args.w_bit and hasattr(config.quant, 'w_qconfig'): 
            config.quant.w_qconfig.bit = args.w_bit
        if args.a_bit and hasattr(config.quant, 'a_qconfig'): 
            config.quant.a_qconfig.bit = args.a_bit
        if args.calib_iter: 
            config.quant.calibrate = args.calib_iter
            
    if hasattr(config, 'process'):
        config.process.seed = args.seed
    return config

# ==============================================================================
# [PATCH] Custom Model Loader for High-Performance Weights
# This function forces loading of official BRECQ weights (71%+) instead of torchvision default
# ==============================================================================
def load_official_model(arch):
    print(f"[*] [Patch] Loading High-Performance pretrained model for {arch}...")
    import torch.hub
    import os
    
    # Check/Create Cache Directory
    cache_dir = os.path.expanduser('~/.cache/torch/checkpoints')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define URLs based on architecture (Using BRECQ's provided weights for fair comparison)
    urls = {
        'resnet18': 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/resnet18_imagenet.pth.tar',
        'resnet50': 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/resnet50_imagenet.pth.tar',
        'mobilenetv2': 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/mobilenetv2.pth.tar'
    }
    
    if arch not in urls:
        print(f"[!] Warning: No official high-performance weight URL for {arch}. Falling back to torchvision.")
        import torchvision.models as models
        return getattr(models, arch)(pretrained=True)

    filename = os.path.basename(urls[arch])
    ckpt_path = os.path.join(cache_dir, filename)

    # 1. Download if missing
    if not os.path.exists(ckpt_path):
        print(f"[*] Downloading {filename} from {urls[arch]}...")
        torch.hub.download_url_to_file(urls[arch], ckpt_path)
    
    # 2. Load Model Architecture (Use QDrop's model definitions for proper block structure)
    from quant_lib.qdrop.model.resnet import resnet18, resnet50
    from quant_lib.qdrop.model.mobilenetv2 import mobilenetv2
    
    if arch == 'mobilenetv2':
        model = mobilenetv2()
    elif arch == 'resnet18':
        model = resnet18()
    elif arch == 'resnet50':
        model = resnet50()
    else:
        # Fallback to torchvision for unsupported models
        import torchvision.models as models
        model = getattr(models, arch)(pretrained=False)
        
    # 3. Load Weights
    print(f"[*] Loading weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Handle state_dict key mismatch if any
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # Remove 'module.' prefix if it exists (common in DDP trained models)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=True)
    print("[*] Successfully loaded High-Performance weights!")
    
    return model

def main():
    # 1. Parse Arguments
    args = get_args()

    # ---------------------------------------------------------
    # [Core Fix] Dynamic Path Loading
    # Logic for path settings based on library structure changes.
    # ---------------------------------------------------------
    root_path = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(root_path, 'quant_lib')

    # [1] Add Common Library Path (quant_lib)
    if lib_path not in sys.path:
        sys.path.append(lib_path)
        print(f"[*] Added common library path: {lib_path}")

    # [2] Add BRECQ Specific Path
    # Note: Even for QDrop/AdaRound, sometimes having this path doesn't hurt, 
    # but strictly speaking only BRECQ needs the internal import fix.
    if args.method == 'brecq':
        brecq_path = os.path.join(lib_path, 'brecq')
        if brecq_path not in sys.path:
            sys.path.append(brecq_path)
        print(f"[*] Added specific library path: {brecq_path} (for BRECQ internal imports)")
    
    # ---------------------------------------------------------

    # 2. Load Config (Smart Loading)
    if args.config is None:
        default_config_path = os.path.join('configs', f'{args.method}.yaml')
        if os.path.exists(default_config_path):
            args.config = default_config_path
            print(f"[*] Auto-loading config: {args.config}")
        elif os.path.exists('config.yaml'): 
            args.config = 'config.yaml'
            print(f"[*] Config specific to {args.method} not found. Using root 'config.yaml'.")
        else:
            print(f"[!] Error: Cannot find configuration file for {args.method}.")
            sys.exit(1)
    
    config = parse_config(args.config)
    config = merge_config(config, args)
    
    # 3. Get Solver (Factory Pattern)
    try:
        SolverClass = get_solver_class(args.method)
    except Exception as e:
        print(f"[!] Solver Loading Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 4. Execution Info
    print(f"[*] Initializing Solver: {args.method}")
    
    w_bit_log = "N/A"
    a_bit_log = "N/A"
    if hasattr(config, 'quant') and hasattr(config.quant, 'w_qconfig'):
        w_bit_log = config.quant.w_qconfig.bit
        a_bit_log = config.quant.a_qconfig.bit
        
    model_name = config.model.type if hasattr(config.model, 'type') else (config.model.name if hasattr(config.model, 'name') else 'Unknown')
    print(f"[*] Model: {model_name}")
    print(f"[*] Settings -> W_Bit: {w_bit_log}, A_Bit: {a_bit_log}")
    
    # 5. Run Solver
    solver = SolverClass(config, args)
    
    # =============================================================
    # [PATCH] Swap Model with Official High-Performance Weights
    # Applies to BRECQ, QDrop, and AdaRound for fair comparison (Start from ~71%)
    # =============================================================
    if args.method in ['brecq', 'qdrop', 'adaround']:
        try:
            # Detect architecture from config
            arch = model_name.lower()
            if 'resnet18' in arch: arch = 'resnet18'
            elif 'resnet50' in arch: arch = 'resnet50'
            elif 'mobile' in arch: arch = 'mobilenetv2'
            
            # Load and swap
            official_model = load_official_model(arch)
            solver.model = official_model.cuda() # Move to GPU
            solver.model.eval()
            print(f"[*] Solver model replaced with official weights for {args.method}.")
        except Exception as e:
            print(f"[!] Failed to swap model: {e}")
            print("[*] Continuing with default model loaded by Solver...")

    # Debug: FP32 Sanity Check=====================================
    print("------------------------------------------------")
    print("[*] Running FP32 Sanity Check (Before Quantization)...")
    from utils.data_utils import validate_model
    # Ensure model is on GPU for validation
    solver.model = solver.model.cuda()
    acc1, acc5 = validate_model(solver.val_loader, solver.model)
    print(f"[*] FP32 Accuracy: Top1 {acc1:.2f}%, Top5 {acc5:.2f}%")
    print("------------------------------------------------")
    # =============================================================
    
    try:
        solver.run()
    except Exception as e:
        print(f"[!] Experiment Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()