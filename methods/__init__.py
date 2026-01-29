# methods/__init__.py

# 나중에 만들 QDropSolver를 미리 import (파일을 아직 안 만들었다면 주석 처리ㄱ)
# from .qdrop_solver import QDropSolver 

def get_solver_class(method_name):
    """
    문자열(method_name)을 받아서 해당 Solver 클래스를 반환
    """
    method_name = method_name.lower()
    
    if method_name == 'qdrop':
        from .qdrop_solver import QDropSolver
        return QDropSolver
        
    elif method_name == 'brecq':
        from .brecq_solver import BRECQSolver
        return BRECQSolver
        
    elif method_name == 'adaround':
        from .adaround_solver import AdaRoundSolver
        return AdaRoundSolver
        
    else:
        raise ValueError(f"Unknown method: {method_name}")