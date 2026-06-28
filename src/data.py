from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset


UNK_TOKEN = "<UNK>"


@dataclass(frozen=True)
class Sentence:
    words: list[str]
    upos: list[str]


def read_conllu(path: str | Path) -> list[Sentence]:
    """Read FORM and UPOS from a CoNLL-U file.

    UPOS is kept only for evaluation. Training code consumes only word ids.
    """
    sentences: list[Sentence] = []
    words: list[str] = []
    upos: list[str] = []

    with Path(path).open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                if words:
                    sentences.append(Sentence(words=words, upos=upos))
                    words = []
                    upos = []
                continue
            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) < 4:
                continue
            token_id = cols[0]
            if "-" in token_id or "." in token_id:
                continue

            words.append(cols[1])
            upos.append(cols[3])

    if words:
        sentences.append(Sentence(words=words, upos=upos))

    return sentences


def build_vocab(sentences: Iterable[Sentence], min_freq: int = 2) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for sent in sentences:
        counts.update(sent.words)

    vocab = {UNK_TOKEN: 0}
    kept = [word for word, count in counts.items() if count >= min_freq]
    kept.sort(key=lambda w: (-counts[w], w))
    for word in kept:
        vocab[word] = len(vocab)
    return vocab


def ids_to_tokens(vocab: dict[str, int]) -> list[str]:
    id_to_token = [""] * len(vocab)
    for token, idx in vocab.items():
        id_to_token[idx] = token
    return id_to_token


def words_to_ids(words: Iterable[str], vocab: dict[str, int]) -> list[int]:
    unk_id = vocab[UNK_TOKEN]
    return [vocab.get(word, unk_id) for word in words]


def vocab_to_json(vocab: dict[str, int], min_freq: int) -> dict[str, object]:
    return {
        "unk_token": UNK_TOKEN,
        "min_freq": min_freq,
        "token_to_id": vocab,
        "id_to_token": ids_to_tokens(vocab),
    }


def vocab_from_json(obj: dict[str, object]) -> dict[str, int]:
    token_to_id = obj["token_to_id"]
    if not isinstance(token_to_id, dict):
        raise ValueError("Invalid vocab JSON: token_to_id must be an object")
    return {str(token): int(idx) for token, idx in token_to_id.items()}


class ConlluDataset(Dataset):
    def __init__(self, sentences: list[Sentence], vocab: dict[str, int]):
        self.sentences = sentences
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sentence = self.sentences[idx]
        return {
            "word_ids": torch.tensor(words_to_ids(sentence.words, self.vocab), dtype=torch.long),
            "words": sentence.words,
            "upos": sentence.upos,
        }


def collate_sentences(batch: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep variable-length sentences as a list."""
    return batch
