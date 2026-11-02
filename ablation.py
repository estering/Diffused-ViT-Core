import argparse

import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the parameter-matched ablation without diffusion loss."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="outputs/without_diffusion_loss")
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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    train.main(parser.parse_args())
