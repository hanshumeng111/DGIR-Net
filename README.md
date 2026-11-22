# DGIR-Net

Evaluation code and pretrained weights for low-light image enhancement. The model is instantiated by the original class name `CIDNet` in `net/CIDNet.py`.

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

| Preset | File under weights/ | Epoch | Inference settings | Training loss weights |
| --- | --- | ---: | --- | --- |
| lolv1 | lolv1.pth | 1300 | Saturation gate on, GT Mean scoring | — |
| lolv2-real | lolv2-real.pth | 1090 | RGB output multiplier 0.8 | Ledge 0.05, Lgate 0.0001 |
| lolv2-syn | lolv2-syn.pth | 120 | Both output gates off | Ledge 0.1, Lgate 0.0005 |

## Results reported in the manuscript

| Dataset | PSNR (dB) | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| LOLv1 | 28.325 | 0.882 | 0.072 |
| LOLv2-Real | 23.607 | 0.863 | 0.128 |
| LOLv2-Synthetic | 26.116 | 0.941 | 0.044 |

## CPU evaluation of the released checkpoints

Full test splits from the [data4LLIE mirror](https://huggingface.co/datasets/apeanut/data4LLIE) were evaluated with the commands above and `--device cpu` using PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu and lpips 0.1.4.

| Dataset | Pairs | PSNR (dB) | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| LOLv1 | 15 | 28.3243 | 0.8821 | 0.0723 |
| LOLv2-Real | 100 | 23.6073 | 0.8635 | 0.1281 |
| LOLv2-Synthetic | 100 | 25.7983 | 0.9362 | 0.0475 |

The synthetic split scored 0.318 dB lower than the manuscript value with the epoch-120 checkpoint. On three images from each LOLv2 split, the released entry point and the original CPU evaluator produced identical 8-bit RGB predictions. The difference on the synthetic split has not been resolved. CUDA execution has not been independently verified in this run.
