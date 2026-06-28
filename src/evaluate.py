from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

from .data import read_conllu
from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate unsupervised clusters against gold UPOS.")
    parser.add_argument("--gold", required=True, help="Gold CoNLL-U file.")
    parser.add_argument("--pred", required=True, help="Predicted CoNLL-U file from src.decode.")
    parser.add_argument("--output", required=True, help="Metrics JSON output path.")
    parser.add_argument("--cluster-summary", default=None, help="Optional TSV summary output path.")
    return parser.parse_args()


def parse_cluster(label: str) -> int:
    if label.startswith("CLUSTER_"):
        return int(label.removeprefix("CLUSTER_"))
    return int(label)


def flatten_aligned(gold_path: str | Path, pred_path: str | Path) -> tuple[list[str], list[int], list[str]]:
    gold_sentences = read_conllu(gold_path)
    pred_sentences = read_conllu(pred_path)
    gold_labels: list[str] = []
    pred_clusters: list[int] = []
    words: list[str] = []

    for sent_idx, (gold_sent, pred_sent) in enumerate(zip_longest(gold_sentences, pred_sentences), start=1):
        if gold_sent is None or pred_sent is None:
            raise ValueError("Gold and prediction sentence counts differ")
        if len(gold_sent.words) != len(pred_sent.words):
            raise ValueError(f"Sentence {sent_idx} token counts differ")
        for tok_idx, (gold_word, pred_word, gold_upos, pred_label) in enumerate(
            zip(gold_sent.words, pred_sent.words, gold_sent.upos, pred_sent.upos),
            start=1,
        ):
            if gold_word != pred_word:
                raise ValueError(
                    f"Token mismatch at sentence {sent_idx}, token {tok_idx}: {gold_word!r} != {pred_word!r}"
                )
            words.append(gold_word)
            gold_labels.append(gold_upos)
            pred_clusters.append(parse_cluster(pred_label))

    return gold_labels, pred_clusters, words


def many_to_one_accuracy(gold_labels: list[str], pred_clusters: list[int]) -> tuple[float, dict[int, str]]:
    cluster_gold_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for gold, cluster in zip(gold_labels, pred_clusters):
        cluster_gold_counts[cluster][gold] += 1

    mapping: dict[int, str] = {}
    correct = 0
    for cluster, counts in cluster_gold_counts.items():
        mapped_pos, mapped_count = counts.most_common(1)[0]
        mapping[cluster] = mapped_pos
        correct += mapped_count
    return correct / max(len(gold_labels), 1), mapping


def entropy(labels: list[object]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return -sum((count / n) * math.log(count / n) for count in counts.values())


def contingency_counts(gold_labels: list[str], pred_clusters: list[int]) -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for gold, cluster in zip(gold_labels, pred_clusters):
        counts[(gold, cluster)] += 1
    return counts


def mutual_information(gold_labels: list[str], pred_clusters: list[int]) -> float:
    n = len(gold_labels)
    if n == 0:
        return 0.0
    gold_counts = Counter(gold_labels)
    cluster_counts = Counter(pred_clusters)
    table = contingency_counts(gold_labels, pred_clusters)
    mi = 0.0
    for (gold, cluster), count in table.items():
        mi += (count / n) * math.log((count * n) / (gold_counts[gold] * cluster_counts[cluster]))
    return mi


def conditional_entropy(target: list[object], given: list[object]) -> float:
    n = len(target)
    if n == 0:
        return 0.0
    given_counts = Counter(given)
    joint_counts: dict[tuple[object, object], int] = defaultdict(int)
    for t, g in zip(target, given):
        joint_counts[(t, g)] += 1
    value = 0.0
    for (t, g), joint in joint_counts.items():
        value -= (joint / n) * math.log(joint / given_counts[g])
    return value


def clustering_metrics(gold_labels: list[str], pred_clusters: list[int]) -> dict[str, float]:
    h_gold = entropy(gold_labels)
    h_cluster = entropy(pred_clusters)
    mi = mutual_information(gold_labels, pred_clusters)
    nmi = 1.0 if h_gold + h_cluster == 0 else (2.0 * mi) / (h_gold + h_cluster)

    homogeneity = 1.0 if h_gold == 0 else 1.0 - conditional_entropy(gold_labels, pred_clusters) / h_gold
    completeness = 1.0 if h_cluster == 0 else 1.0 - conditional_entropy(pred_clusters, gold_labels) / h_cluster
    v_measure = 0.0 if homogeneity + completeness == 0 else 2.0 * homogeneity * completeness / (homogeneity + completeness)

    return {
        "v_measure": v_measure,
        "nmi": nmi,
        "ari": adjusted_rand_index(gold_labels, pred_clusters),
    }


def comb2(n: int) -> float:
    return n * (n - 1) / 2.0


def adjusted_rand_index(gold_labels: list[str], pred_clusters: list[int]) -> float:
    n = len(gold_labels)
    if n < 2:
        return 1.0

    table = contingency_counts(gold_labels, pred_clusters)
    gold_counts = Counter(gold_labels)
    cluster_counts = Counter(pred_clusters)

    sum_comb = sum(comb2(count) for count in table.values())
    sum_gold = sum(comb2(count) for count in gold_counts.values())
    sum_cluster = sum(comb2(count) for count in cluster_counts.values())
    total = comb2(n)
    expected = (sum_gold * sum_cluster) / total if total else 0.0
    max_index = 0.5 * (sum_gold + sum_cluster)
    denom = max_index - expected
    return 1.0 if denom == 0 else (sum_comb - expected) / denom


def write_cluster_summary(
    path: str | Path,
    gold_labels: list[str],
    pred_clusters: list[int],
    words: list[str],
    mapping: dict[int, str],
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    cluster_word_counts: dict[int, Counter[str]] = defaultdict(Counter)
    cluster_gold_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for word, gold, cluster in zip(words, gold_labels, pred_clusters):
        cluster_word_counts[cluster][word] += 1
        cluster_gold_counts[cluster][gold] += 1

    with path.open("w", encoding="utf-8") as f:
        f.write("cluster_id\tsize\tmapped_pos\ttop_words\tgold_pos_distribution\n")
        for cluster in sorted(cluster_gold_counts):
            gold_counts = cluster_gold_counts[cluster]
            size = sum(gold_counts.values())
            top_words = ", ".join(word for word, _ in cluster_word_counts[cluster].most_common(10))
            pos_dist = ", ".join(f"{pos}:{count / size:.2f}" for pos, count in gold_counts.most_common())
            f.write(f"{cluster}\t{size}\t{mapping.get(cluster, '_')}\t{top_words}\t{pos_dist}\n")


def main() -> None:
    args = parse_args()
    gold_labels, pred_clusters, words = flatten_aligned(args.gold, args.pred)
    mto, mapping = many_to_one_accuracy(gold_labels, pred_clusters)
    metrics = {
        "many_to_one_accuracy": mto,
        **clustering_metrics(gold_labels, pred_clusters),
        "num_tokens": len(gold_labels),
        "num_clusters_used": len(set(pred_clusters)),
    }

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    summary_path = args.cluster_summary
    if summary_path is None:
        summary_path = output_path.with_name("cluster_summary.tsv")
    write_cluster_summary(summary_path, gold_labels, pred_clusters, words, mapping)


if __name__ == "__main__":
    main()
