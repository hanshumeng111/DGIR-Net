# DGIR-Net

Evaluation code and pretrained weights for low-light image enhancement. The model is instantiated as `DGIRNet` from `net/DGIRNet.py`. Checkpoints saved before the DGDI rename are supported automatically.

## Setup

Install a matched CUDA-enabled PyTorch and torchvision pair for your GPU. From this repository run:

```bash
python -m pip install -r requirements-eval.txt
python -c "import torch; print(torch.cuda.is_available())"
```

The second command should print `True` for GPU testing. On the first paired evaluation, LPIPS needs torchvision's pretrained AlexNet weights in the torch hub cache or network access to download them.

## Evaluate

Point the two directory arguments to the actual folders containing the low-light images and their reference images. Each output directory must be new or empty. Benchmark data are supplied separately by the user.

```bash
python eval_gpu.py --dataset lolv1 --low-dir /data/LOL/eval15/low --gt-dir /data/LOL/eval15/high --output-dir results/lolv1
python eval_gpu.py --dataset lolv2-real --low-dir /data/LOLv2/Real_captured/Test/Low --gt-dir /data/LOLv2/Real_captured/Test/Normal --output-dir results/lolv2-real
python eval_gpu.py --dataset lolv2-syn --low-dir /data/LOLv2/Synthetic/Test/Low --gt-dir /data/LOLv2/Synthetic/Test/Normal --output-dir results/lolv2-syn
```

The default device is `cuda:0`. Use `--device cuda:1` to select another GPU. Omit `--gt-dir` for inference only. Each preset automatically selects the matching checkpoint. Predictions are written as RGB PNG files. Paired tests print the number of matched images and the mean of per-image PSNR, RGB SSIM and AlexNet LPIPS. The LOLv1 GT Mean correction is applied to the 8-bit predictions for scoring only. The saved predictions do not use reference data.

| Preset | File under weights/  |
| --- | --- |
| lolv1 | lolv1.pth |
| lolv2-real | lolv2-real.pth| 
| lolv2-syn | lolv2-syn.pth | 

## Results reported in the manuscript

| Dataset | PSNR (dB) | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| LOLv1 | 28.325 | 0.882 | 0.072 |
| LOLv2-Real | 23.607 | 0.863 | 0.128 |
| LOLv2-Synthetic | 26.116 | 0.941 | 0.044 |

