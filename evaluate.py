import argparse

import torch

from diffused_vit.config import DiffusionConfig, ModelConfig
from diffused_vit.data import build_loader
from diffused_vit.metrics import calculate_binary_metrics
from diffused_vit.model import DiffusedViT
from diffused_vit.training import get_device, predict


def main(args):
    device = get_device(args.device)
    model = DiffusedViT(
        ModelConfig(dropout=args.dropout, pretrained=False),
        DiffusionConfig(
            timesteps=args.timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
        ),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    loader = build_loader(args.data_dir, False, args.batch_size)
    labels, scores = predict(model, loader, device)
    print(calculate_binary_metrics(labels, scores, args.threshold).to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.6442)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--device", default="auto")
    main(parser.parse_args())
