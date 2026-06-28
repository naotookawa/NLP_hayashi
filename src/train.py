from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import ConlluDataset, build_vocab, collate_sentences, ids_to_tokens, read_conllu, vocab_to_json
from .features import build_feature_vocab, make_word_feature_tensor
from .model import GenerativeHMM
from .utils import append_jsonl, ensure_dir, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a generative HMM for unsupervised word clustering.")
    parser.add_argument("--train", required=True, help="Training CoNLL-U file.")
    parser.add_argument("--dev", default=None, help="Optional development CoNLL-U file.")
    parser.add_argument("--num-tags", type=int, default=17, help="Number of latent clusters.")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum training frequency for vocabulary entries.")
    parser.add_argument("--use-features", action="store_true", help="Use feature-corrected p(x|z) emissions.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sentence_nlls(model: GenerativeHMM, batch: list[dict[str, object]], device: torch.device) -> tuple[torch.Tensor, int]:
    log_initial = model.log_initial_probs()
    log_transition = model.log_transition_probs()
    log_emission = model.log_emission_probs()

    losses = []
    token_count = 0
    for item in batch:
        word_ids = item["word_ids"]
        if not isinstance(word_ids, torch.Tensor):
            raise TypeError("word_ids must be a torch.Tensor")
        word_ids = word_ids.to(device)
        losses.append(-model.sequence_log_prob_from_params(word_ids, log_initial, log_transition, log_emission))
        token_count += int(word_ids.numel())
    return torch.stack(losses), token_count


@torch.no_grad()
def evaluate_nll(
    model: GenerativeHMM,
    dataset: ConlluDataset,
    batch_size: int,
    device: torch.device,
    show_progress: bool = False,
    desc: str = "dev",
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sentences)
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_sentences = 0
    iterator = tqdm(
        loader,
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=False,
        disable=not show_progress,
    )
    for batch in iterator:
        losses, token_count = sentence_nlls(model, batch, device)
        total_nll += float(losses.sum().item())
        total_tokens += token_count
        total_sentences += len(batch)
        iterator.set_postfix(nll_per_token=f"{total_nll / max(total_tokens, 1):.4f}", refresh=False)
    return {
        "nll": total_nll,
        "nll_per_token": total_nll / max(total_tokens, 1),
        "num_tokens": float(total_tokens),
        "num_sentences": float(total_sentences),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    log_path = output_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    train_sentences = read_conllu(args.train)
    if not train_sentences:
        raise ValueError(f"No training sentences found in {args.train}")
    dev_sentences = read_conllu(args.dev) if args.dev else []

    vocab = build_vocab(train_sentences, min_freq=args.min_freq)
    id_to_token = ids_to_tokens(vocab)

    word_feature_tensor = None
    feature_to_id: dict[str, int] = {}
    word_feature_ids: list[list[int]] = []
    if args.use_features:
        feature_to_id, word_feature_ids = build_feature_vocab(id_to_token)
        word_feature_tensor = make_word_feature_tensor(word_feature_ids)

    model = GenerativeHMM(
        num_tags=args.num_tags,
        vocab_size=len(vocab),
        word_feature_ids=word_feature_tensor,
        num_features=len(feature_to_id),
        init_scale=args.init_scale,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_dataset = ConlluDataset(train_sentences, vocab)
    dev_dataset = ConlluDataset(dev_sentences, vocab) if dev_sentences else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sentences,
    )

    config = {
        "train": args.train,
        "dev": args.dev,
        "num_tags": args.num_tags,
        "min_freq": args.min_freq,
        "use_features": args.use_features,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "init_scale": args.init_scale,
        "grad_clip": args.grad_clip,
        "vocab_size": len(vocab),
        "num_features": len(feature_to_id),
        "progress": not args.no_progress,
    }
    save_json(config, output_dir / "config.json")
    save_json(vocab_to_json(vocab, args.min_freq), output_dir / "vocab.json")

    best_metric = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_nll = 0.0
        total_tokens = 0
        total_sentences = 0

        train_iterator = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{args.epochs} train",
            unit="batch",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=True,
            disable=args.no_progress,
        )
        for batch in train_iterator:
            optimizer.zero_grad()
            losses, token_count = sentence_nlls(model, batch, device)
            loss = losses.mean()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_nll += float(losses.sum().item())
            total_tokens += token_count
            total_sentences += len(batch)
            train_iterator.set_postfix(
                loss=f"{float(loss.item()):.4f}",
                nll_per_token=f"{total_nll / max(total_tokens, 1):.4f}",
                refresh=False,
            )

        record: dict[str, float | int | None] = {
            "epoch": epoch,
            "train_nll": total_nll,
            "train_nll_per_token": total_nll / max(total_tokens, 1),
            "train_num_tokens": total_tokens,
            "train_num_sentences": total_sentences,
            "dev_nll": None,
            "dev_nll_per_token": None,
        }

        selection_metric = record["train_nll_per_token"]
        if dev_dataset is not None:
            dev_stats = evaluate_nll(
                model,
                dev_dataset,
                args.batch_size,
                device,
                show_progress=not args.no_progress,
                desc=f"epoch {epoch}/{args.epochs} dev",
            )
            record["dev_nll"] = dev_stats["nll"]
            record["dev_nll_per_token"] = dev_stats["nll_per_token"]
            selection_metric = dev_stats["nll_per_token"]

        append_jsonl(record, log_path)
        tqdm.write(
            "epoch {epoch}: train_nll/token={train:.4f} dev_nll/token={dev}".format(
                epoch=epoch,
                train=float(record["train_nll_per_token"]),
                dev=(
                    f"{float(record['dev_nll_per_token']):.4f}"
                    if record["dev_nll_per_token"] is not None
                    else "n/a"
                ),
            )
        )

        if float(selection_metric) < best_metric:
            best_metric = float(selection_metric)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    model.load_state_dict(best_state)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config | {"best_epoch": best_epoch},
        "vocab": vocab_to_json(vocab, args.min_freq),
        "feature_to_id": feature_to_id,
        "word_feature_ids": word_feature_ids,
    }
    torch.save(checkpoint, output_dir / "model.pt")


if __name__ == "__main__":
    main()
