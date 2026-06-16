import argparse
import json
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from protsae.data import load_activation_data, load_normal_forms, normal_forms_to_tensors
from protsae.model import ProtSAE
from protsae.torch_utils import FastTensorDataLoader
from protsae.esm import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train ProtSAE on ESM-2 activations.")
    parser.add_argument("--data-root", default="deepgo_dataset")
    parser.add_argument("--activation-root", default="activation_dataset")
    parser.add_argument("--output-dir", default="final_result")
    parser.add_argument("--ont", choices=["bp", "cc", "mf"], default="bp")
    parser.add_argument("--layer", type=int, default=35)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--num-features", type=int, default=None)
    parser.add_argument("--input-dim", type=int, default=5120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lambda-annot", type=float, default=10.0)
    parser.add_argument("--lambda-axiom", type=float, default=1.0)
    parser.add_argument("--save-after-epoch", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def default_num_features(ont):
    return 40000 if ont == "bp" else 30000


def normalized_reconstruction_error(x, x_hat):
    L_rec = torch.sum((x_hat - x) ** 2, dim=-1).mean()
    baseline = torch.sum((x - x.mean(dim=0, keepdim=True)) ** 2, dim=-1).mean()
    return L_rec / baseline


def evaluate_validation(model, dataloader, normal_forms, lambda_annot, lambda_axiom, device):
    model.eval()
    stats = {
        "L_total": 0.0,
        "L_rec": 0.0,
        "L_annot": 0.0,
        "L_axiom": 0.0,
        "normalized_L_rec": 0.0,
        "L0": 0.0,
    }

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss, loss_dict = model.loss(
                x=x,
                y=y,
                normal_forms=normal_forms,
                lambda_annot=lambda_annot,
                lambda_axiom=lambda_axiom,
                output=output,
            )
            del loss

            stats["L_total"] += loss_dict["L_total"].item()
            stats["L_rec"] += loss_dict["L_rec"].item()
            stats["L_annot"] += loss_dict["L_annot"].item()
            stats["L_axiom"] += loss_dict["L_axiom"].item()
            stats["normalized_L_rec"] += normalized_reconstruction_error(x, output.x_hat).item()
            stats["L0"] += (output.z > 0).float().sum(dim=-1).mean().item()

    num_batches = len(dataloader)
    for key in stats:
        stats[key] /= num_batches
    return stats


def main():
    args = parse_args()
    set_random_seed(args.seed)

    dtype = torch.float32
    if args.num_features is None:
        args.num_features = default_num_features(args.ont)

    (train_data, valid_data, terms_dict) = load_activation_data(
        data_root=args.data_root,
        activation_root=args.activation_root,
        ont=args.ont,
        layer=args.layer,
        dtype=dtype,
    )
    num_concepts = len(terms_dict)

    nf1, nf2, nf3, nf4, relations, zero_concepts = load_normal_forms(
        go_norm_file=Path(args.data_root) / "go.norm",
        terms_dict=terms_dict,
    )
    normal_forms = normal_forms_to_tensors((nf1, nf2, nf3, nf4), args.device)

    model = ProtSAE(
        input_dim=args.input_dim,
        num_concepts=num_concepts,
        num_features=args.num_features,
        top_k=args.top_k,
        num_zero_concepts=len(zero_concepts),
        num_relations=len(relations),
        dtype=dtype,
        seed=args.seed,
        device=args.device,
    ).to(args.device)

    train_loader = FastTensorDataLoader(
        train_data[0], train_data[1], batch_size=args.batch_size, shuffle=False
    )
    valid_loader = FastTensorDataLoader(
        valid_data[0], valid_data[1], batch_size=args.batch_size, shuffle=False
    )

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"protsae_{args.ont}_layer{args.layer}_topk{args.top_k}"
    checkpoint_path = output_dir / f"{run_name}.pt"
    log_path = output_dir / f"{run_name}.json"

    training_log = []
    best_score = None
    best_epoch = None

    for epoch in range(args.epochs):
        model.train()
        epoch_stats = {
            "L_total": 0.0,
            "L_rec": 0.0,
            "L_annot": 0.0,
            "L_axiom": 0.0,
            "normalized_L_rec": 0.0,
        }

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False):
            x = x.to(args.device)
            y = y.to(args.device)

            output = model(x)
            loss, loss_dict = model.loss(
                x=x,
                y=y,
                normal_forms=normal_forms,
                lambda_annot=args.lambda_annot,
                lambda_axiom=args.lambda_axiom,
                output=output,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_stats["L_total"] += loss_dict["L_total"].item()
            epoch_stats["L_rec"] += loss_dict["L_rec"].item()
            epoch_stats["L_annot"] += loss_dict["L_annot"].item()
            epoch_stats["L_axiom"] += loss_dict["L_axiom"].item()
            epoch_stats["normalized_L_rec"] += normalized_reconstruction_error(
                x, output.x_hat
            ).item()

        scheduler.step()

        for key in epoch_stats:
            epoch_stats[key] /= len(train_loader)
        valid_stats = evaluate_validation(
            model=model,
            dataloader=valid_loader,
            normal_forms=normal_forms,
            lambda_annot=args.lambda_annot,
            lambda_axiom=args.lambda_axiom,
            device=args.device,
        )

        training_log.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in epoch_stats.items()},
                **{f"valid_{key}": value for key, value in valid_stats.items()},
            }
        )

        if epoch >= args.save_after_epoch:
            score = (valid_stats["L_annot"], valid_stats["normalized_L_rec"])
            if best_score is None or score < best_score:
                best_score = score
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_path)

    if not checkpoint_path.exists():
        torch.save(model.state_dict(), checkpoint_path)
        best_epoch = args.epochs - 1

    with open(log_path, "w") as handle:
        json.dump(
            {
                "config": {
                    "data_root": args.data_root,
                    "activation_root": args.activation_root,
                    "ont": args.ont,
                    "layer": args.layer,
                    "top_k": args.top_k,
                    "num_features": args.num_features,
                    "input_dim": args.input_dim,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "lambda_annot": args.lambda_annot,
                    "lambda_axiom": args.lambda_axiom,
                    "seed": args.seed,
                },
                "checkpoint": str(checkpoint_path),
                "best_epoch": best_epoch,
                "history": training_log,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
