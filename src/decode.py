from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import read_conllu, vocab_from_json, words_to_ids
from .features import make_word_feature_tensor
from .model import GenerativeHMM
from .utils import ensure_dir, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode latent clusters with a trained generative HMM.")
    parser.add_argument("--model", required=True, help="Path to model.pt.")
    parser.add_argument("--input", required=True, help="Input CoNLL-U file.")
    parser.add_argument("--output", required=True, help="Output CoNLL-U predictions.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_model(path: str | Path, device: torch.device) -> tuple[GenerativeHMM, dict[str, int]]:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    vocab = vocab_from_json(checkpoint["vocab"])

    word_feature_ids = checkpoint.get("word_feature_ids") or []
    feature_to_id = checkpoint.get("feature_to_id") or {}
    word_feature_tensor = make_word_feature_tensor(word_feature_ids) if feature_to_id else None

    model = GenerativeHMM(
        num_tags=int(config["num_tags"]),
        vocab_size=len(vocab),
        word_feature_ids=word_feature_tensor,
        num_features=len(feature_to_id),
        init_scale=float(config.get("init_scale", 0.01)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab


def write_predictions(sentences: list[tuple[list[str], list[int]]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        for words, tags in sentences:
            for idx, (word, tag) in enumerate(zip(words, tags), start=1):
                cols = [
                    str(idx),
                    word,
                    "_",
                    f"CLUSTER_{tag}",
                    "_",
                    "_",
                    "_",
                    "_",
                    "_",
                    "_",
                ]
                f.write("\t".join(cols))
                f.write("\n")
            f.write("\n")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    model, vocab = load_model(args.model, device)

    decoded: list[tuple[list[str], list[int]]] = []
    with torch.no_grad():
        log_initial = model.log_initial_probs()
        log_transition = model.log_transition_probs()
        log_emission = model.log_emission_probs()
    for sentence in read_conllu(args.input):
        word_ids = torch.tensor(words_to_ids(sentence.words, vocab), dtype=torch.long, device=device)
        tags = model.viterbi_decode_from_params(word_ids, log_initial, log_transition, log_emission)
        decoded.append((sentence.words, tags))

    write_predictions(decoded, args.output)


if __name__ == "__main__":
    main()
