import argparse
from pathlib import Path

import torch

from diffused_vit.config import DiffusionConfig, ModelConfig
from diffused_vit.data import build_loader
from diffused_vit.model import DiffusedViT
from diffused_vit.training import get_device, run_epoch, set_seed


def main(args):
    set_seed(args.seed)
    device = get_device(args.device)
    data_root = Path(args.data_root)
    train_loader = build_loader(
        data_root / "train", True, args.batch_size, args.workers
    )
    val_loader = build_loader(
        data_root / "val", False, args.batch_size, args.workers
    )
    model = DiffusedViT(
        ModelConfig(dropout=args.dropout, pretrained=not args.no_pretrained),
        DiffusionConfig(
            timesteps=args.timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.lr_decay
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            epoch,
            args.epochs,
            args.diffusion_weight,
            optimizer,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            device,
            epoch,
            args.epochs,
            args.diffusion_weight,
        )
        scheduler.step()
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model": model.state_dict()}, output_dir / "best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1.19e-5)
    parser.add_argument("--lr-decay", type=float, default=0.977)
    parser.add_argument("--diffusion-weight", type=float, default=0.221)
    parser.add_argument("--dropout", type=float, default=0.6442)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    main(parser.parse_args())
