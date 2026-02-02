---
name: capstone benchmark issue template
about: capstone issue template about benchmarking
title: "[Experiment] title"
labels: 'Experiment'
assignees: ''
---

## Experiment Objective



## Configuration

| Parameter/Setting | Value | Description |
|:------|:------|:------|
| Model Architecture | ResNet-50(Model Name) | 여기엔 가중치 파일 이름 또는 path ex. path/to/fp32_baseline.pth
| Dataset | ImageNet_val | .... |
| Precision | W4A8 | Weight 4bit, Activation 8bit |
| Calibration Method(weight) | Min-Max, Percentile | ... |
| Calibration Method(activation) | Min-Max, Percentile | ... |
| Quantization Granularity | Per-channel / per-tensor | ...|
| Device | RTX 3080ti | 실험 수행 디바이스 |

## Execution Command
```sh



```

## Expexted Results

- Baseline: 
- Target Accuracy: 