from __future__ import annotations

import torch
from torch import nn

from .features import PAD_FEATURE_ID


class GenerativeHMM(nn.Module):
    """A generative HMM for unsupervised POS-like word clustering.

    Emissions are always p(x | z): each hidden tag owns a distribution over
    the vocabulary, normalized with log_softmax(..., dim=1).
    """

    def __init__(
        self,
        num_tags: int,
        vocab_size: int,
        word_feature_ids: torch.Tensor | None = None,
        num_features: int = 0,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.vocab_size = vocab_size
        self.use_features = word_feature_ids is not None and num_features > 0

        self.initial_logits = nn.Parameter(torch.randn(num_tags) * init_scale)
        self.transition_logits = nn.Parameter(torch.randn(num_tags, num_tags) * init_scale)
        self.emission_logits = nn.Parameter(torch.randn(num_tags, vocab_size) * init_scale)

        if self.use_features:
            if word_feature_ids is None:
                raise ValueError("word_feature_ids is required when num_features > 0")
            self.feature_weights = nn.Parameter(torch.randn(num_features, num_tags) * init_scale)
            self.register_buffer("word_feature_ids", word_feature_ids.long())
        else:
            self.feature_weights = None
            self.register_buffer("word_feature_ids", torch.empty((vocab_size, 0), dtype=torch.long))

    def log_initial_probs(self) -> torch.Tensor:
        return torch.log_softmax(self.initial_logits, dim=0)

    def log_transition_probs(self) -> torch.Tensor:
        # log_transition[j, k] = log p(z_t = k | z_{t-1} = j)
        return torch.log_softmax(self.transition_logits, dim=1)

    def emission_scores(self) -> torch.Tensor:
        """Return unnormalized score(k, v) before vocabulary softmax."""
        scores = self.emission_logits
        if not self.use_features:
            return scores

        feature_ids = self.word_feature_ids
        mask = feature_ids.ne(PAD_FEATURE_ID)
        safe_ids = feature_ids.clamp_min(0)
        feature_scores = self.feature_weights[safe_ids] * mask.unsqueeze(-1)
        # Sum feature weights for each word, then transpose [V, K] -> [K, V].
        feature_scores_by_word = feature_scores.sum(dim=1).transpose(0, 1)
        return scores + feature_scores_by_word

    def log_emission_probs(self) -> torch.Tensor:
        # log_emission[k, v] = log p(x_t = v | z_t = k)
        # Normalizing over dim=1 keeps one vocabulary distribution per tag.
        return torch.log_softmax(self.emission_scores(), dim=1)

    def sequence_log_prob_from_params(
        self,
        word_ids: torch.Tensor,
        log_initial: torch.Tensor,
        log_transition: torch.Tensor,
        log_emission: torch.Tensor,
    ) -> torch.Tensor:
        """Forward algorithm for log p(x), fully marginalizing z."""
        if word_ids.numel() == 0:
            raise ValueError("Cannot score an empty sentence")

        emissions = log_emission[:, word_ids]

        # alpha[0, k] = log p(z_1=k) + log p(x_1 | z_1=k)
        alpha = log_initial + emissions[:, 0]

        for t in range(1, word_ids.numel()):
            # alpha[t, k] = log p(x_t | z_t=k)
            #             + logsumexp_j alpha[t-1, j] + log p(k | j)
            transition_scores = alpha.unsqueeze(1) + log_transition
            alpha = emissions[:, t] + torch.logsumexp(transition_scores, dim=0)

        # log p(x) = logsumexp_k alpha[T-1, k]
        return torch.logsumexp(alpha, dim=0)

    def sequence_log_prob(self, word_ids: torch.Tensor) -> torch.Tensor:
        """Forward algorithm for log p(x), fully marginalizing z."""
        return self.sequence_log_prob_from_params(
            word_ids,
            self.log_initial_probs(),
            self.log_transition_probs(),
            self.log_emission_probs(),
        )

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        return self.sequence_log_prob(word_ids)

    def neg_log_likelihood(self, word_ids: torch.Tensor) -> torch.Tensor:
        return -self.sequence_log_prob(word_ids)

    @torch.no_grad()
    def viterbi_decode_from_params(
        self,
        word_ids: torch.Tensor,
        log_initial: torch.Tensor,
        log_transition: torch.Tensor,
        log_emission: torch.Tensor,
    ) -> list[int]:
        """Viterbi algorithm for argmax_z p(z | x)."""
        if word_ids.numel() == 0:
            return []

        emissions = log_emission[:, word_ids]

        # delta[0, k] = log p(z_1=k) + log p(x_1 | z_1=k)
        delta = log_initial + emissions[:, 0]
        backpointers: list[torch.Tensor] = []

        for t in range(1, word_ids.numel()):
            # delta[t, k] = log p(x_t | z_t=k)
            #             + max_j delta[t-1, j] + log p(k | j)
            transition_scores = delta.unsqueeze(1) + log_transition
            best_scores, best_prev = torch.max(transition_scores, dim=0)
            delta = emissions[:, t] + best_scores
            backpointers.append(best_prev)

        last_tag = int(torch.argmax(delta).item())
        tags = [last_tag]
        for backpointer in reversed(backpointers):
            last_tag = int(backpointer[last_tag].item())
            tags.append(last_tag)
        tags.reverse()
        return tags

    @torch.no_grad()
    def viterbi_decode(self, word_ids: torch.Tensor) -> list[int]:
        """Viterbi algorithm for argmax_z p(z | x)."""
        return self.viterbi_decode_from_params(
            word_ids,
            self.log_initial_probs(),
            self.log_transition_probs(),
            self.log_emission_probs(),
        )
