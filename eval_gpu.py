"""Evaluate the released DGIR-Net checkpoints on paired LOL benchmarks."""
import argparse
from pathlib import Path
import re

DATASETS = ("lolv1", "lolv2-real", "lolv2-syn")
ROOT = Path(__file__).resolve().parent
FORMATS = {".png", ".bmp", ".jpg", ".jpeg"}


def image_index(folder):
    files = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in FORMATS:
            continue
        # LOLv2 distributions can use low00001 / normal00001 for one pair.
        key = re.sub(r"^(low|normal|high)[_-]?(?=\d+$)", "", path.stem.lower())
        if key in files:
            raise ValueError(f"Ambiguous image ID {key!r}: {files[key]} and {path}")
        files[key] = path
    if not files:
        raise ValueError(f"No input images in {folder}")
    return files


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--low-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", type=Path, help="RGB references for paired evaluation")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty directory")
    parser.add_argument("--weights", type=Path, help="Override the dataset checkpoint")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 by default; cpu for diagnostics")
    return parser.parse_args()


def run(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms.functional import to_pil_image, to_tensor
    from net.DGIRNet import DGIRNet

    low = image_index(args.low_dir)
    gt = image_index(args.gt_dir) if args.gt_dir else None
    if gt is not None and low.keys() != gt.keys():
        raise ValueError(f"Unmatched IDs: missing GT {sorted(low.keys()-gt.keys())[:10]}, "
                         f"missing inputs {sorted(gt.keys()-low.keys())[:10]}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Choose a new or empty output directory: {args.output_dir}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install a CUDA-enabled PyTorch build or use --device cpu.")
    checkpoint = args.weights or ROOT / "weights" / f"{args.dataset}.pth"
    model = DGIRNet()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().to(device)
    model.trans.gated = args.dataset == "lolv1"
    model.trans.gated2 = args.dataset == "lolv2-real"
    model.trans.alpha = 0.8 if model.trans.gated2 else 1.0
    if gt is not None:
        import lpips
        from paired_metrics import score
        perceptual = lpips.LPIPS(net="alex").eval().to(device)
    totals = np.zeros(3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.dataset}: {len(low)} images | {checkpoint.name} | {device} | "
          f"alpha={model.trans.alpha} | GT Mean={gt is not None and model.trans.gated}")
    with torch.inference_mode():
        for key, path in low.items():
            with Image.open(path) as image:
                x = to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
            h, w = x.shape[-2:]
            # Three downsampling stages. Official LOL image sizes need no padding.
            if h % 8 or w % 8:
                x = F.pad(x, (0, -w % 8, 0, -h % 8), mode="replicate")
            prediction = model(x)[0, :, :h, :w].clamp(0, 1)
            restored = to_pil_image(prediction.cpu())
            restored.save(args.output_dir / (path.stem + ".png"))
            if gt is not None:
                with Image.open(gt[key]) as image:
                    reference = image.convert("RGB")
                if restored.size != reference.size:
                    raise ValueError(f"Input/reference dimensions differ for {key}")
                totals += score(np.asarray(restored), np.asarray(reference), perceptual,
                                device, gt_mean=model.trans.gated)
    if gt is not None:
        psnr, ssim, perceptual_score = totals / len(low)
        print(f"{len(low)} pairs | PSNR {psnr:.4f} dB | SSIM {ssim:.4f} | LPIPS {perceptual_score:.4f}")


if __name__ == "__main__":
    run(arguments())
