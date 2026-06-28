from __future__ import annotations

import string
import unicodedata
from collections import Counter
from typing import Iterable

import torch

from .data import UNK_TOKEN


PAD_FEATURE_ID = -1


def is_punctuation(word: str) -> bool:
    if not word:
        return False
    return all((ch in string.punctuation) or unicodedata.category(ch).startswith("P") for ch in word)


def length_bucket(word: str) -> str:
    length = len(word)
    if length <= 1:
        return "1"
    if length <= 3:
        return "2-3"
    if length <= 6:
        return "4-6"
    if length <= 10:
        return "7-10"
    return "11+"


def extract_features(word: str) -> list[str]:
    if word == UNK_TOKEN:
        return ["special=<UNK>"]

    lower = word.lower()
    feats = [f"lower={lower}", f"len={length_bucket(word)}"]

    for n in (1, 2, 3):
        if len(lower) >= n:
            feats.append(f"prefix{n}={lower[:n]}")
            feats.append(f"suffix{n}={lower[-n:]}")

    if word[:1].isupper():
        feats.append("is_capitalized")
    if word.isupper() and any(ch.isalpha() for ch in word):
        feats.append("is_all_upper")
    if any(ch.isdigit() for ch in word):
        feats.append("contains_digit")
    if "-" in word:
        feats.append("contains_hyphen")
    if is_punctuation(word):
        feats.append("is_punctuation")

    return feats


def build_feature_vocab(
    id_to_token: Iterable[str],
    min_feature_count: int = 1,
) -> tuple[dict[str, int], list[list[int]]]:
    tokens = list(id_to_token)
    counts: Counter[str] = Counter()
    per_word_features: list[list[str]] = []
    for token in tokens:
        feats = extract_features(token)
        per_word_features.append(feats)
        counts.update(feats)

    kept = [feat for feat, count in counts.items() if count >= min_feature_count]
    kept.sort()
    feature_to_id = {feat: idx for idx, feat in enumerate(kept)}

    word_feature_ids: list[list[int]] = []
    for feats in per_word_features:
        ids = [feature_to_id[feat] for feat in feats if feat in feature_to_id]
        word_feature_ids.append(ids)

    return feature_to_id, word_feature_ids


def make_word_feature_tensor(word_feature_ids: list[list[int]]) -> torch.Tensor:
    max_len = max((len(ids) for ids in word_feature_ids), default=0)
    if max_len == 0:
        return torch.empty((len(word_feature_ids), 0), dtype=torch.long)

    tensor = torch.full((len(word_feature_ids), max_len), PAD_FEATURE_ID, dtype=torch.long)
    for word_idx, ids in enumerate(word_feature_ids):
        if ids:
            tensor[word_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return tensor
