"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

from . import positional_encoders, caches

from .utils import LayerNorm
debug = False
def print_deb(string):
    if debug:
        print(string)

RECALL_CACHE_INTERVENTIONS = frozenset(
    {
        "target-value-corruption",
        "target-repeat-only",
        "target-repeat-masked",
    }
)

RECALL_ATTENTION_MASK_INTERVENTIONS = frozenset(
    {
        "target-repeat-only",
        "target-repeat-masked",
    }
)

class InPlaceSetSlice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, full_tensor, last_slice, x_val, dim):

        if last_slice is None:
            prev_length = 0
        else:
            prev_length = last_slice.shape[dim]
        new_length = prev_length + x_val.shape[dim]

        prefix_slice = [slice(None)] * dim # for the first 'dim' dimensions we want to take everything
        full_tensor[prefix_slice + [slice(prev_length, new_length)]] = x_val
        ctx.prev_length = prev_length
        ctx.new_length = new_length
        ctx.dim = dim
        ret = torch.Tensor().to(full_tensor)
        ret.set_(full_tensor[prefix_slice +[slice(None,new_length)]])
        return ret

    @staticmethod
    def backward(ctx, grad_out):
        prefix_slice = [slice(None)] * ctx.dim 
        if ctx.prev_length == 0:
            return None, None, grad_out[prefix_slice + [slice(None, ctx.new_length)]], None
        else:
            return None, grad_out[prefix_slice + [slice(None, ctx.prev_length)]], grad_out[prefix_slice + [slice(ctx.prev_length, ctx.new_length)]], None


def apply_inplace_set(x_acc, x_val, dim):
    full_tensor, last_slice = x_acc
    new_slice = InPlaceSetSlice.apply(full_tensor, last_slice, x_val, dim)
    return full_tensor, new_slice


class CausalSelfAttention(nn.Module): # should be able to use bidirectional as well in the future not now though

    def __init__(self, config, lm_cache):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias) # cotf models use fused QKV and BUT uses seperate Q and fused KV 
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.cache_storage = lm_cache.get_storage_for_layer(self)
        self.config = config
        self.allow_cache_during_training = getattr(config, "allow_cache_during_training", False)
        self.attention_mode = getattr(config, "attention_mode", "causal")
        if self.attention_mode not in ("causal","bidirectional"):
            raise ValueError(f"Unsupported attention mode: {self.attention_mode}")
        self.is_causal = self.attention_mode == "causal"
        self.attention_implementation = getattr(
            config, "attention_implementation", "sdpa"
        )
        if self.attention_implementation not in ("sdpa", "manual"):
            raise ValueError(
                "Unsupported attention implementation: "
                f"{self.attention_implementation}"
            )


        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if self.flash:
            assert config.attention_window_length is None
        else:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
        bias = torch.tril(torch.ones(config.sequence_length, config.sequence_length))
        if config.attention_window_length is not None:
            bias = torch.triu(bias, diagonal=-config.attention_window_length)
        self.register_buffer("bias", bias.view(1, 1, config.sequence_length, config.sequence_length))

        self.drop_cache()


    @staticmethod
    def apply_recall_cache_intervention_mask(
        attention_logits,
        *,
        intervention,
        target_repeat,
        evolution_repeats,
        recall_index,
        tokens_per_repeat,
    ):
        """Apply an explicit repeat-block mask before recall softmax."""
        if intervention not in RECALL_ATTENTION_MASK_INTERVENTIONS:
            raise ValueError(
                f"Unsupported recall attention intervention: {intervention!r}."
            )

        if attention_logits.ndim != 4:
            raise RuntimeError(
                "Recall masking expects attention logits with shape [B, H, Q, K]."
            )

        key_tokens = int(attention_logits.shape[-1])
        tokens_per_repeat = int(tokens_per_repeat)

        if key_tokens % tokens_per_repeat != 0:
            raise RuntimeError(
                "Attention keys do not contain whole repeat blocks."
            )

        cached_repeats = key_tokens // tokens_per_repeat
        expected_cached_repeats = int(evolution_repeats) + int(recall_index)

        if cached_repeats != expected_cached_repeats:
            raise RuntimeError(
                "Unexpected recall cache depth: "
                f"expected {expected_cached_repeats} blocks, "
                f"found {cached_repeats}."
            )

        target_start = (int(target_repeat) - 1) * tokens_per_repeat
        target_end = target_start + tokens_per_repeat
        evolution_token_end = int(evolution_repeats) * tokens_per_repeat

        if target_start < 0 or target_end > evolution_token_end:
            raise RuntimeError(
                "The requested target block is outside the evolution cache."
            )

        masked_keys = torch.zeros(
            key_tokens,
            dtype=torch.bool,
            device=attention_logits.device,
        )

        if intervention == "target-repeat-only":
            # Hide all evolution blocks, then expose the requested one.
            # Recall-phase blocks, including the current block, remain visible.
            masked_keys[:evolution_token_end] = True
            masked_keys[target_start:target_end] = False

        elif intervention == "target-repeat-masked":
            # Hide only the requested evolution block.
            masked_keys[target_start:target_end] = True

        intervened_logits = attention_logits.masked_fill(
            masked_keys.view(1, 1, 1, -1),
            float("-inf"),
        )

        if not torch.isfinite(intervened_logits).any(dim=-1).all():
            raise RuntimeError(
                "Recall intervention left at least one query with no visible keys."
            )

        return intervened_logits
    @staticmethod
    def _summarize_attention(
        attention_logits,
        attention_probabilities,
        *,
        queries,
        keys,
        values,
        tokens_per_repeat,
        recent_window=None,
        attention_mask=None,
    ):
        B, H, Q, K = attention_probabilities.shape
        diagnostics={}

        if K % tokens_per_repeat != 0:
            raise ValueError(
                "Cached key length must contain whole repeat blocks."
            )
        print_deb(f"attention proba contiguous: {attention_probabilities.is_contiguous()}")
        cached_repeats = K // tokens_per_repeat
        diagnostics['cached_repeats'] = cached_repeats
        finite_logits = torch.isfinite(attention_logits)
        visible_token_count = finite_logits.sum(dim=-1) # B,H,Q
        visible_repeat_mask = finite_logits.reshape(
            B, H, Q, cached_repeats, tokens_per_repeat
        ).any(dim=-1) # B,H,Q,R
        visible_repeat_count = visible_repeat_mask.sum(dim=-1) # B,H,Q
        diagnostics['visible_token_count'] = visible_token_count
        diagnostics['visible_repeat_count'] = visible_repeat_count
        print_deb(f"cached reps {cached_repeats}")
        print_deb(f"k over c = {K//cached_repeats}")
        # rerepeat_masses = attention_probabilities.view(B,H,Q,cached_repeats,Q)

        # print(f"rerepeat_masses mass: {rerepeat_masses.is_contiguous()}")
        attention_by_repeat = attention_probabilities.reshape(
            B, H, Q, cached_repeats, tokens_per_repeat
        )
        #print(f"attention by repeat shape {attention_by_repea t.shape}")
        repeat_mass = attention_by_repeat.sum(dim=-1) #B,H,T,R
        diagnostics['repeat_mass'] = repeat_mass
        #print(f"repeat mass shape {repeat_mass.shape}")
        # per_head_repeat_mass = repeat_mass()
        head_repeat_mass = repeat_mass.mean(dim=(0,2)) # H,R
        diagnostics['head_repeat_mass'] = head_repeat_mass
        macro_repeat_mass = head_repeat_mass.mean(dim=0) # R
        diagnostics['macro_repeat_mass'] = macro_repeat_mass
        # macro_head_repeat_mass = head_repeat_mass.mean(dim=1) # H
        repeat_entropy=-torch.xlogy(repeat_mass,repeat_mass).sum(dim=-1)
        diagnostics['repeat_entropy'] = repeat_entropy
        #print(f"repeat entropy = {repeat_entropy.shape}")
        repeat_entropy_denominator = visible_repeat_count.to(
            repeat_entropy.dtype
        ).clamp_min(1).log()
        normalised_repeat_entropy = torch.where(
            visible_repeat_count > 1,
            repeat_entropy / repeat_entropy_denominator.clamp_min(
                torch.finfo(repeat_entropy.dtype).tiny
            ),
            torch.zeros_like(repeat_entropy),
        )
        diagnostics['normalised_repeat_entropy'] = normalised_repeat_entropy
        head_repeat_entropy = repeat_entropy.mean(dim=(0,2)) # H
        diagnostics['head_repeat_entropy'] = head_repeat_entropy
        # macro_repeat_entropy = head_repeat_entropy.mean(dim=0) #r

        local_probabilities = attention_by_repeat / repeat_mass.unsqueeze(
            -1
        ).clamp_min(torch.finfo(attention_probabilities.dtype).tiny)
        within_repeat_entropy = -torch.xlogy(
            local_probabilities, local_probabilities
        ).sum(dim=-1) # B,H,Q,R
        diagnostics['within_repeat_entropy'] = within_repeat_entropy
        head_within_repeat_entropy = within_repeat_entropy.mean(dim=(0,2)) # H,R
        diagnostics['head_within_repeat_entropy'] = head_within_repeat_entropy
        diagnostics['macro_within_repeat_entropy'] = (
            head_within_repeat_entropy.mean(dim=0)
        ) # R

        if Q != tokens_per_repeat:
            raise ValueError(
                "Same-position attention requires one query per spatial token."
            )
        same_position_mass = torch.diagonal(
            attention_by_repeat, dim1=2, dim2=4
        ).transpose(-1, -2) # B,H,Q,R
        diagnostics['same_position_mass'] = same_position_mass
        head_same_position_mass = same_position_mass.mean(dim=(0,2)) # H,R
        diagnostics['head_same_position_mass'] = head_same_position_mass
        diagnostics['macro_same_position_mass'] = (
            head_same_position_mass.mean(dim=0)
        ) # R

        selected_query_indices = sorted({0, Q // 2, Q - 1})
        selected_query_tensor = torch.tensor(
            selected_query_indices,
            device=attention_probabilities.device,
            dtype=torch.long,
        )
        diagnostics['selected_query_indices'] = selected_query_indices
        selected_query_attention_per_example = attention_by_repeat.index_select(
            2, selected_query_tensor
        ) # B,H,selected_queries,R,T
        diagnostics['selected_query_attention_per_example'] = (
            selected_query_attention_per_example
        )
        diagnostics['selected_query_attention'] = (
            selected_query_attention_per_example.mean(dim=0)
        ) # H,selected_queries,R,T

        finite_logit_count = finite_logits.sum(dim=-1)
        safe_logit_count = finite_logit_count.clamp_min(1)
        finite_logit_values = torch.where(
            finite_logits, attention_logits, torch.zeros_like(attention_logits)
        )
        logit_mean = finite_logit_values.sum(dim=-1) / safe_logit_count
        centred_logits = torch.where(
            finite_logits,
            attention_logits - logit_mean.unsqueeze(-1),
            torch.zeros_like(attention_logits),
        )
        logit_std = torch.sqrt(
            centred_logits.square().sum(dim=-1) / safe_logit_count
        )
        logit_min = attention_logits.masked_fill(
            ~finite_logits, float('inf')
        ).amin(dim=-1)
        logit_max = attention_logits.masked_fill(
            ~finite_logits, float('-inf')
        ).amax(dim=-1)
        logit_statistics = {
            'logit_min': logit_min,
            'logit_max': logit_max,
            'logit_mean': logit_mean,
            'logit_std': logit_std,
            'logit_spread': logit_max - logit_min,
        }
        for metric_name, metric_value in logit_statistics.items():
            diagnostics[metric_name] = metric_value # B,H,Q
            diagnostics[f'head_{metric_name}'] = metric_value.mean(dim=(0,2)) # H

        if K >= 2:
            top_two_logits = attention_logits.masked_fill(
                ~finite_logits, float('-inf')
            ).topk(k=2, dim=-1).values
            top_two_logit_gap = top_two_logits[..., 0] - top_two_logits[..., 1]
            top_two_logit_gap_valid = finite_logit_count >= 2
            top_two_logit_gap = torch.where(
                top_two_logit_gap_valid,
                top_two_logit_gap,
                torch.zeros_like(top_two_logit_gap),
            )
        else:
            top_two_logit_gap = attention_logits.new_zeros((B, H, Q))
            top_two_logit_gap_valid = torch.zeros(
                (B, H, Q), device=attention_logits.device, dtype=torch.bool
            )
        diagnostics['top_two_logit_gap'] = top_two_logit_gap
        diagnostics['top_two_logit_gap_valid'] = top_two_logit_gap_valid
        valid_gap_count = top_two_logit_gap_valid.sum(dim=(0,2)).clamp_min(1)
        diagnostics['head_top_two_logit_gap'] = (
            top_two_logit_gap.sum(dim=(0,2)) / valid_gap_count
        ) # H

        attention_entropy = -torch.xlogy(
            attention_probabilities, attention_probabilities
        ).sum(dim=-1) # B,H,Q
        diagnostics['attention_entropy'] = attention_entropy
        diagnostics['head_attention_entropy'] = attention_entropy.mean(dim=(0,2))
        diagnostics['macro_attention_entropy'] = attention_entropy.mean()
        attention_entropy_denominator = visible_token_count.to(
            attention_entropy.dtype
        ).clamp_min(1).log()
        normalised_attention_entropy = torch.where(
            visible_token_count > 1,
            attention_entropy / attention_entropy_denominator.clamp_min(
                torch.finfo(attention_entropy.dtype).tiny
            ),
            torch.zeros_like(attention_entropy),
        )
        diagnostics['normalised_attention_entropy'] = normalised_attention_entropy
        diagnostics['head_normalised_attention_entropy'] = (
            normalised_attention_entropy.mean(dim=(0,2))
        )
        diagnostics['macro_normalised_attention_entropy'] = (
            normalised_attention_entropy.mean()
        )

        effective_support = torch.exp(attention_entropy)
        diagnostics['effective_support'] = effective_support
        diagnostics['head_effective_support'] = effective_support.mean(dim=(0,2))
        diagnostics['macro_effective_support'] = effective_support.mean()
        effective_support_fraction = effective_support / visible_token_count.to(
            effective_support.dtype
        ).clamp_min(1)
        diagnostics['effective_support_fraction'] = effective_support_fraction
        diagnostics['head_effective_support_fraction'] = (
            effective_support_fraction.mean(dim=(0,2))
        )
        diagnostics['macro_effective_support_fraction'] = (
            effective_support_fraction.mean()
        )

        maximum_attention_probability = attention_probabilities.amax(dim=-1)
        diagnostics['maximum_attention_probability'] = maximum_attention_probability
        diagnostics['head_maximum_attention_probability'] = (
            maximum_attention_probability.mean(dim=(0,2))
        )
        diagnostics['macro_maximum_attention_probability'] = (
            maximum_attention_probability.mean()
        )

        logits_by_repeat = attention_logits.reshape(
            B, H, Q, cached_repeats, tokens_per_repeat
        )
        raw_repeat_logsumexp = torch.logsumexp(
            logits_by_repeat, dim=-1
        ) # B,H,Q,R
        repeat_logsumexp_valid = visible_repeat_mask
        repeat_logsumexp = torch.where(
            repeat_logsumexp_valid,
            raw_repeat_logsumexp,
            torch.zeros_like(raw_repeat_logsumexp),
        )
        diagnostics['repeat_logsumexp'] = repeat_logsumexp
        diagnostics['repeat_logsumexp_valid'] = repeat_logsumexp_valid
        head_repeat_logsumexp_count = repeat_logsumexp_valid.sum(
            dim=(0,2)
        )
        diagnostics['head_repeat_logsumexp'] = repeat_logsumexp.sum(
            dim=(0,2)
        ) / head_repeat_logsumexp_count.clamp_min(1)
        diagnostics['head_repeat_logsumexp_valid'] = (
            head_repeat_logsumexp_count > 0
        )
        macro_repeat_logsumexp_count = repeat_logsumexp_valid.sum(
            dim=(0,1,2)
        )
        diagnostics['macro_repeat_logsumexp'] = repeat_logsumexp.sum(
            dim=(0,1,2)
        ) / macro_repeat_logsumexp_count.clamp_min(1)
        diagnostics['macro_repeat_logsumexp_valid'] = (
            macro_repeat_logsumexp_count > 0
        )

        recent_repeat_count = 1 if recent_window is None else int(recent_window)
        if recent_repeat_count <= 0:
            raise ValueError("recent_window must be positive when provided.")
        recent_repeat_count = min(recent_repeat_count, cached_repeats)
        diagnostics['recent_repeat_count'] = recent_repeat_count
        if recent_repeat_count < cached_repeats:
            recent_logsumexp = torch.logsumexp(
                raw_repeat_logsumexp[..., -recent_repeat_count:], dim=-1
            )
            old_logsumexp = torch.logsumexp(
                raw_repeat_logsumexp[..., :-recent_repeat_count], dim=-1
            )
            recent_vs_old_margin_valid = torch.isfinite(
                recent_logsumexp
            ) & torch.isfinite(old_logsumexp)
            raw_recent_vs_old_margin = recent_logsumexp - old_logsumexp
            recent_vs_old_margin = torch.where(
                recent_vs_old_margin_valid,
                raw_recent_vs_old_margin,
                torch.zeros_like(raw_recent_vs_old_margin),
            )
            diagnostics['recent_vs_old_logsumexp_margin'] = recent_vs_old_margin
            diagnostics['recent_vs_old_logsumexp_margin_valid'] = (
                recent_vs_old_margin_valid
            )
            head_margin_count = recent_vs_old_margin_valid.sum(dim=(0,2))
            diagnostics['head_recent_vs_old_logsumexp_margin'] = (
                recent_vs_old_margin.sum(dim=(0,2))
                / head_margin_count.clamp_min(1)
            )
            diagnostics['head_recent_vs_old_logsumexp_margin_valid'] = (
                head_margin_count > 0
            )
            macro_margin_count = recent_vs_old_margin_valid.sum()
            diagnostics['macro_recent_vs_old_logsumexp_margin'] = (
                recent_vs_old_margin.sum() / macro_margin_count.clamp_min(1)
            )
            diagnostics['macro_recent_vs_old_logsumexp_margin_valid'] = (
                macro_margin_count > 0
            )
        else:
            diagnostics['recent_vs_old_logsumexp_margin'] = None
            diagnostics['recent_vs_old_logsumexp_margin_valid'] = None
            diagnostics['head_recent_vs_old_logsumexp_margin'] = None
            diagnostics['head_recent_vs_old_logsumexp_margin_valid'] = None
            diagnostics['macro_recent_vs_old_logsumexp_margin'] = None
            diagnostics['macro_recent_vs_old_logsumexp_margin_valid'] = None

        repeat_ages = torch.arange(
            cached_repeats - 1,
            -1,
            -1,
            device=attention_probabilities.device,
        )
        age_band_labels = ['current', '1-2', '3-4', '5-8', 'older_than_8']
        age_band_masks = (
            repeat_ages == 0,
            (repeat_ages >= 1) & (repeat_ages <= 2),
            (repeat_ages >= 3) & (repeat_ages <= 4),
            (repeat_ages >= 5) & (repeat_ages <= 8),
            repeat_ages > 8,
        )
        age_band_mass = torch.stack(
            [
                repeat_mass[..., band_mask].sum(dim=-1)
                for band_mask in age_band_masks
            ],
            dim=-1,
        ) # B,H,Q,age_band
        diagnostics['repeat_age_band_labels'] = age_band_labels
        diagnostics['repeat_age_band_mass'] = age_band_mass
        diagnostics['head_repeat_age_band_mass'] = age_band_mass.mean(dim=(0,2))
        diagnostics['macro_repeat_age_band_mass'] = age_band_mass.mean(dim=(0,1,2))

        head_jensen_shannon_per_example = attention_probabilities.new_zeros(
            (B, H, H)
        )
        probability_floor = torch.finfo(attention_probabilities.dtype).tiny
        for left_head in range(H):
            for right_head in range(left_head + 1, H):
                left_probability = attention_probabilities[:, left_head]
                right_probability = attention_probabilities[:, right_head]
                midpoint_probability = 0.5 * (
                    left_probability + right_probability
                )
                midpoint_probability = midpoint_probability.clamp_min(
                    probability_floor
                )
                divergence = 0.5 * (
                    torch.xlogy(
                        left_probability,
                        left_probability / midpoint_probability,
                    ).sum(dim=-1)
                    + torch.xlogy(
                        right_probability,
                        right_probability / midpoint_probability,
                    ).sum(dim=-1)
                )
                per_example_divergence = divergence.mean(dim=-1)
                head_jensen_shannon_per_example[:, left_head, right_head] = (
                    per_example_divergence
                )
                head_jensen_shannon_per_example[:, right_head, left_head] = (
                    per_example_divergence
                )
        diagnostics['head_jensen_shannon_divergence_per_example'] = (
            head_jensen_shannon_per_example
        )
        head_jensen_shannon = head_jensen_shannon_per_example.mean(dim=0)
        diagnostics['head_jensen_shannon_divergence'] = head_jensen_shannon
        diagnostics['normalised_head_jensen_shannon_divergence'] = (
            head_jensen_shannon / math.log(2)
        )

        if queries.shape[:3] != (B, H, Q):
            raise ValueError("Query shape does not match attention probabilities.")
        if keys.shape[:3] != (B, H, K):
            raise ValueError("Key shape does not match attention probabilities.")
        if values.shape[:3] != (B, H, K):
            raise ValueError("Value shape does not match attention probabilities.")
        if not (queries.shape[-1] == keys.shape[-1] == values.shape[-1]):
            raise ValueError("Query, key, and value head dimensions must match.")

        query_norm = torch.linalg.vector_norm(queries, dim=-1) # B,H,Q
        diagnostics['query_norm'] = query_norm
        diagnostics['head_query_norm_mean'] = query_norm.mean(dim=(0,2))
        diagnostics['head_query_norm_std'] = query_norm.std(
            dim=(0,2), unbiased=False
        )

        head_dimension = keys.shape[-1]
        cached_vectors = {
            'key': keys.reshape(
                B, H, cached_repeats, tokens_per_repeat, head_dimension
            ),
            'value': values.reshape(
                B, H, cached_repeats, tokens_per_repeat, head_dimension
            ),
        }
        for vector_name, vectors_by_repeat in cached_vectors.items():
            vector_norm = torch.linalg.vector_norm(
                vectors_by_repeat, dim=-1
            ) # B,H,R,T
            diagnostics[f'{vector_name}_norm'] = vector_norm
            diagnostics[f'head_{vector_name}_norm_mean'] = vector_norm.mean(
                dim=(0,3)
            ) # H,R
            diagnostics[f'head_{vector_name}_norm_std'] = vector_norm.std(
                dim=(0,3), unbiased=False
            ) # H,R

        block_contribution = torch.einsum(
            'bhqrt,bhrtd->bhqrd',
            attention_by_repeat,
            cached_vectors['value'],
        )
        block_contribution_norm = torch.linalg.vector_norm(
            block_contribution, dim=-1
        ) # B,H,Q,R
        diagnostics['block_contribution_norm'] = block_contribution_norm
        diagnostics['head_block_contribution_norm_mean'] = (
            block_contribution_norm.mean(dim=(0,2))
        ) # H,R
        diagnostics['head_block_contribution_norm_std'] = (
            block_contribution_norm.std(dim=(0,2), unbiased=False)
        ) # H,R
        diagnostics['macro_block_contribution_norm'] = (
            block_contribution_norm.mean(dim=(0,1,2))
        ) # R

        # macro_head_repeat_entopy = macro_head_repeat_mass.mean(dim=-1) #H







        print_deb(f"repeat masses shape {repeat_mass.shape}")



        return diagnostics
    def init_cache(self, expected_total_length):
        self._lazy_init_cache_length = expected_total_length
    @torch.no_grad()
    def permute_cached_target_values(
        self,
        *,
        target_repeat,
        tokens_per_repeat,
        permutation_offset,
    ):
        """Replace one cached repeat's values with another batch example's values."""
        if not isinstance(self.all_values, tuple) or len(self.all_values) != 2:
            raise RuntimeError(
                "Unexpected CoTFormer value-cache representation."
            )

        full_values, populated_values = self.all_values
        if populated_values is None:
            raise RuntimeError(
                "Value cache is empty before the recall intervention."
            )

        batch_size = int(full_values.shape[0])
        if batch_size < 2:
            raise ValueError(
                "Target-value corruption requires batches containing "
                "at least two examples."
            )

        tokens_per_repeat = int(tokens_per_repeat)
        target_repeat = int(target_repeat)

        target_start = (target_repeat - 1) * tokens_per_repeat
        target_end = target_start + tokens_per_repeat

        if target_start < 0 or target_end > populated_values.shape[2]:
            raise RuntimeError(
                "The requested evolution block has not been populated."
            )

        offset = int(permutation_offset) % batch_size
        if offset == 0:
            offset = 1

        permutation = torch.roll(
            torch.arange(batch_size, device=full_values.device),
            shifts=offset,
        )

        replacement = full_values[
            :, :, target_start:target_end, :
        ].index_select(0, permutation).clone()

        full_values[:, :, target_start:target_end, :] = replacement

        return {
            "batch_size": batch_size,
            "permutation_offset": offset,
            "target_cache_block_zero_based": target_repeat - 1,
            "target_token_start": target_start,
            "target_token_end_exclusive": target_end,
        }
    def drop_cache(self):
        self.all_keys = None
        self.all_values = None
        self.all_indices = None
        self._lazy_init_cache_length = None
        

    def forward(self, x, pos_emb_closure, cache_context, start_index, indices,collect_attention_diagnostics=False, 
                attention_diagnostics_recent_window=None,repeat_cache_window=None,
                recall_cache_intervention=None,
                recall_target_repeat=None,
                evolution_repeats=None,
                recall_index=None): # indices seem to be a leftover from ACT variants, unused in fixed depth models
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        C = self.n_embd
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        diag = None
        

        pos_size = k.shape[-1] // 2

        q = pos_emb_closure.adapt_queries(q, start_index=start_index, indices=indices)
        if cache_context is not None and self.cache_storage is not None:
            att_prefix, cache_values_dict = \
                self.cache_storage.retrieve_for_query(q, cache_context, pos_emb_closure, start_index)
            if self.training and att_prefix is not None and not self.allow_cache_during_training:
                raise ValueError("Cache is not allowed during training")
        else:
            att_prefix = None
        k_before_pos = k
        k = pos_emb_closure.adapt_keys(k, start_index=start_index, indices=indices)

        if self._lazy_init_cache_length is not None:
            # assert indices is not None
            self.all_keys = (
                k.new_empty((B, self.n_head, self._lazy_init_cache_length, C // self.n_head)), #  allocate cache with the same head dimension
                None
            )
            self.all_values = (
                v.new_empty((B, self.n_head, self._lazy_init_cache_length, C // self.n_head)),
                None
            )
            self._lazy_init_cache_length = None
        
        if self.all_keys is not None:
            self.all_keys = apply_inplace_set(self.all_keys, k, dim=2) # write all heads along the repeat axis, B,H,hs untouched
            self.all_values = apply_inplace_set(self.all_values, v, dim=2)
            k = self.all_keys[1]
            v = self.all_values[1]  
            print_deb(f"k is {k.shape} v is {v.shape}")          

            if self.is_causal:

            # assert indices is not None
                k = self.all_keys[1]
                v = self.all_values[1]
                attn_mask = self.bias[:,:,:T,:T].unsqueeze(3).repeat(
                    1, 1, 1, k.shape[2] // T, 1
                ).unsqueeze(0).view(1, 1, q.shape[2], k.shape[2]) == 1
            else:
                attn_mask = None
            sdpa_is_causal = False

        else:
            attn_mask = None
            sdpa_is_causal = self.is_causal
        
        if (
            self.flash
            and self.attention_implementation == "sdpa"
            and collect_attention_diagnostics == False
        ):
            if att_prefix is not None:
                raise NotImplementedError
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            attn_mask=attn_mask,
                            dropout_p=self.dropout if self.training else 0.0,
                            is_causal=sdpa_is_causal,         # should be false when using bidirectional AND CAUSAL
                        )
        else:
            # manual implementation of attention
            print_deb("MANUAL ATT")
            print_deb("MANUAL ATT")
            print_deb("MANUAL ATT")
            print_deb("MANUAL ATT")
            print_deb("MANUAL ATT")
            num_k = k.shape[2]
            num_q = q.shape[-1]
            # print(f"k shape is {num_k} and q shape is {num_q} from inside causal attention else man att")
            current_reps = num_k // T
            if repeat_cache_window is not None and repeat_cache_window < current_reps:
                cutoff = (current_reps-repeat_cache_window) * T
                excluded_key_mask = torch.zeros((1,1,1,num_k),dtype=torch.bool,device=k.device)
                excluded_key_mask[..., :cutoff] = True
                




            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if repeat_cache_window is not None and repeat_cache_window < current_reps:
                att = att.masked_fill(excluded_key_mask,float("-inf"))
            # att = pos_emb_closure.adapt_attention_before_softmax(att, start_query_index=start_index, start_key_index=start_index)
            if attn_mask is None and self.is_causal:
                attn_mask = self.bias[:,:,:T,:T] == 1
            if attn_mask is not None:

                att = att.masked_fill(~attn_mask, float('-inf'))
            if att_prefix is not None:
                prefix_size = att_prefix.shape[-1]
                current_size = att.shape[-1]
                att = torch.cat((att_prefix, att), dim=-1)
            if recall_cache_intervention is not None:
                if att_prefix is not None:
                    raise RuntimeError(
                        "Recall cache interventions do not support "
                        "external LM-cache prefixes."
                    )

                att = self.apply_recall_cache_intervention_mask(
                    att,
                    intervention=recall_cache_intervention,
                    target_repeat=recall_target_repeat,
                    evolution_repeats=evolution_repeats,
                    recall_index=recall_index,
                    tokens_per_repeat=T,
                )

            # if collect_attention_diagnostics:
            #     att_logits = att

            # att = F.softmax(att, dim=-1)
            if collect_attention_diagnostics==True:

                att_logits = att

            att = F.softmax(att, dim=-1)
            if collect_attention_diagnostics==True:

                att_proba = att
            att = self.attn_dropout(att)
            if att_prefix is not None:
                att_prefix, att = torch.split(att, (prefix_size, current_size), dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            if att_prefix is not None:
                cache_v = cache_values_dict['v']
                if cache_v.ndim == v.ndim:
                    y += att_prefix @ cache_v
                elif cache_v.ndim == v.ndim + 1:
                    y += (att_prefix.unsqueeze(3) @ cache_v).squeeze(3)
                else:
                    raise NotImplementedError
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        if cache_context is not None and self.cache_storage is not None:
            with torch.no_grad():
                self.cache_storage.store_in_cache(k_before_pos, {'v': v})
        if collect_attention_diagnostics:
            diag = self._summarize_attention(
                att_logits,
                att_proba,
                queries=q,
                keys=k,
                values=v,
                tokens_per_repeat=T,
                recent_window=attention_diagnostics_recent_window,
                attention_mask=attn_mask,
            )
        return y, diag

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config, lm_cache):
        super().__init__()
        self.config = config
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, lm_cache)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self,x,pos_emb_closure,cache_context,
                start_index,indices=None,
                collect_attention_diagnostics=False,
                attention_diagnostics_recent_window=None,
                repeat_cache_window=None,
                recall_cache_intervention=None,
                recall_target_repeat=None,
                evolution_repeats=None,
                recall_index=None):
        attention_output, diagnostics = self.attn(
            self.ln_1(x),
            pos_emb_closure,
            cache_context,
            start_index,
            indices,
            collect_attention_diagnostics=collect_attention_diagnostics,
            attention_diagnostics_recent_window=(
                attention_diagnostics_recent_window
            ),
            repeat_cache_window=repeat_cache_window,
            recall_cache_intervention=recall_cache_intervention,
            recall_target_repeat=recall_target_repeat,
            evolution_repeats=evolution_repeats,
            recall_index=recall_index,
        )


        x = x + attention_output
        x = x + self.mlp(self.ln_2(x))

        return x, diagnostics

class ForwardBackwardEmbedding(nn.Module):
    FORWARD = 0
    RECALL = 1

    def __init__(self, config):
        super().__init__()
        self.embd = nn.Embedding(2, config.n_embd)

    def forward(self, direction_ids):
        # [B] -> [B, D] -> [B, 1, D]
        return self.embd(direction_ids).unsqueeze(1)
class RelativeAgeEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embd = nn.Embedding(
            config.ca_max_relative_age + 1,
            config.n_embd,
        )

    def forward(self, age_ids):
        # [B] -> [B, D] -> [B, 1, D]
        return self.embd(age_ids).unsqueeze(1)

class GPTBase(nn.Module):

    needs_iter = False

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.n_repeat = config.n_repeat
        self.direction_embedding = ForwardBackwardEmbedding(config)
        self.age_embedding = RelativeAgeEmbedding(config)

        self.controller_application = config.ca_controller_application
        self.lm_cache = caches.get_cache(config.lm_cache)(config)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = positional_encoders.get_encoder(config.positional_encoder)(config),
            drop = nn.Dropout(config.dropout),
            h_begin = nn.ModuleList( 
                [Block(config, self.lm_cache) for _ in range(config.n_layer_begin)]
            ),
            h_mid = nn.ModuleList(
                [Block(config, self.lm_cache) 
                for _ in range(config.n_layer_begin, config.n_layer - config.n_layer_end)],
            ),
            h_end = nn.ModuleList( 
                [Block(config, self.lm_cache) 
                for _ in range(config.n_layer - config.n_layer_end, config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        def _post_init_fn(module):
            if hasattr(module, "post_init"):
                module.post_init()
        self.apply(_post_init_fn)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= sum(p.numel() for p in self.transformer.wpe.parameters()) # TODO: Why do we need this?
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,
                 idx,
                   targets=None,
                     get_logits=False, 
                     use_cache=False, 
                     iter=None, 
                return_all_logits=False,
                num_repeats=None,
                return_repeat_states=False,
                return_repeat_logits=False,
                delayed_recall=False,
                recall_age=None,
                return_attention_diagnostics=False,
                attention_diagnostics_recent_window=None,
                repeat_cache_window=None,
                intervention_source_depth=None,
                intervention_input_ids=None,
                cache_reset_source_depth=None,
                num_recall_repeats =1,
                recall_cache_intervention=None,
                recall_value_permutation_offset=1):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        repeats = self.n_repeat if num_repeats is None else int(num_repeats)
        if repeats <= 0:
            raise ValueError("num_repeats must be positive.")
        
        # if delayed_recall:
        #     if recall_age is None:
        #         raise ValueError(
        #             "recall_age is required for delayed recall."
        #         )

        #     if not 0 <= recall_age < repeats:
        #         raise ValueError(
        #             f"recall_age must be between 0 and {repeats - 1}."
        #         )
        recall_repeats = int(num_recall_repeats)

        if delayed_recall:
            if recall_age is None:
                raise ValueError(
                    "recall_age is required for delayed recall."
                )
            if recall_repeats <= 0:
                raise ValueError("num_recall_repeats must be positive.")
            
        recall_target_repeat = None

        if delayed_recall:
            if isinstance(recall_age, bool) or not isinstance(recall_age, int):
                raise TypeError("recall_age must be an integer.")

            if not 0 <= recall_age < repeats:
                raise ValueError(
                    f"recall_age must be between 0 and {repeats - 1}."
                )

            recall_target_repeat = repeats - recall_age

        if recall_cache_intervention is not None:
            if recall_cache_intervention not in RECALL_CACHE_INTERVENTIONS:
                raise ValueError(
                    "Unsupported recall cache intervention: "
                    f"{recall_cache_intervention!r}."
                )
            
            if not delayed_recall:
                raise ValueError(
                    "Recall cache interventions require delayed_recall=True."
                )

            if self.training:
                raise ValueError(
                    "Recall cache interventions are evaluation-only."
                )

     

     
            if repeat_cache_window is not None:
                raise ValueError(
                    "Recall cache interventions require the full-cache policy."
                )

            if (
                recall_cache_intervention
                in RECALL_ATTENTION_MASK_INTERVENTIONS
                and any(
                    block.attn.attention_implementation != "manual"
                    for block in self.transformer.h_mid
                )
            ):
                raise ValueError(
                    "Recall attention masking requires manual attention."
                )

            if (
                isinstance(recall_value_permutation_offset, bool)
                or not isinstance(recall_value_permutation_offset, int)
                or recall_value_permutation_offset <= 0
            ):
                raise ValueError(
                    "recall_value_permutation_offset must be a positive integer."
                )


        if repeats > self.config.ca_max_relative_age:
            raise ValueError(
                f"num_repeats={repeats} exceeds "
                f"ca_max_relative_age={self.config.ca_max_relative_age}."
            )
        def make_ids(value):
            return torch.full(
                (b,),
                value,
                device=idx.device,
                dtype=torch.long,
            )


        def make_controller(direction, relative_age):
            direction_ids = make_ids(direction)
            age_ids = make_ids(relative_age)

            direction_embedding = self.direction_embedding(
                direction_ids
            )

            age_embedding = self.age_embedding(
                age_ids
            )

            return direction_embedding + age_embedding
        has_intervention_depth = intervention_source_depth is not None
        has_intervention_inputs = intervention_input_ids is not None
        if has_intervention_depth != has_intervention_inputs:
            raise ValueError(
                "intervention_source_depth and intervention_input_ids must be "
                "supplied together."
            )
        if has_intervention_depth:
            if self.training:
                raise ValueError(
                    "Clean-state intervention is only supported in evaluation mode."
                )
            if use_cache:
                raise ValueError(
                    "Clean-state intervention does not support external LM cache mode."
                )
            if isinstance(intervention_source_depth, bool) or not isinstance(
                intervention_source_depth, int
            ):
                raise TypeError("intervention_source_depth must be an integer.")
            if not 0 <= intervention_source_depth < repeats:
                raise ValueError(
                    "intervention_source_depth must be in the range "
                    "0 .. num_repeats - 1."
                )
            if len(self.transformer.h_begin) != 0:
                raise ValueError(
                    "Clean-state intervention currently requires n_layer_begin == 0."
                )
            if not isinstance(intervention_input_ids, torch.Tensor):
                raise TypeError("intervention_input_ids must be a torch.Tensor.")
            if intervention_input_ids.shape != idx.shape:
                raise ValueError(
                    "intervention_input_ids must have the same shape as the "
                    f"original input: expected {tuple(idx.shape)}, got "
                    f"{tuple(intervention_input_ids.shape)}."
                )
            if intervention_input_ids.device != idx.device:
                raise ValueError(
                    "intervention_input_ids must be on the same device as the "
                    "original input."
                )
            if intervention_input_ids.dtype != idx.dtype:
                raise TypeError(
                    "intervention_input_ids must have the same dtype as the "
                    "original input."
                )
        has_cache_reset = cache_reset_source_depth is not None
        if has_cache_reset:
            if self.training:
                raise ValueError(
                    "Cache reset is only supported in evaluation mode."
                )
            if use_cache:
                raise ValueError(
                    "Cache reset does not support external LM cache mode."
                )
            if isinstance(cache_reset_source_depth, bool) or not isinstance(
                cache_reset_source_depth, int
            ):
                raise TypeError("cache_reset_source_depth must be an integer.")
            if not 0 <= cache_reset_source_depth < repeats:
                raise ValueError(
                    "cache_reset_source_depth must be in the range "
                    "0 .. num_repeats - 1."
                )
            if (
                has_intervention_depth
                and cache_reset_source_depth != intervention_source_depth
            ):
                raise ValueError(
                    "State intervention and cache reset must use the same source depth."
                )
        if recall_cache_intervention is not None:
            if has_intervention_depth or has_cache_reset:
                raise ValueError(
                "Recall cache interventions cannot be combined with "
                "clean-state or cache-reset interventions.")
        if return_attention_diagnostics and self.training:
            raise ValueError(
                "Attention diagnostics are only supported in evaluation mode."
            )
        if repeat_cache_window is not None:
            if isinstance(repeat_cache_window, bool) or not isinstance(
                repeat_cache_window, int
            ):
                raise TypeError(
                    "repeat_cache_window must be a positive integer or None."
                )
            if repeat_cache_window <= 0:
                raise ValueError("repeat_cache_window must be positive.")
            # if self.training:
            #     raise ValueError(
            #         "repeat_cache_window is evaluation-only."
            #     )
        if repeat_cache_window is not None or has_cache_reset:
            if self.config.attention_mode != "bidirectional":
                raise ValueError(
                    "repeat_cache_window currently requires bidirectional attention."
                )
            if self.config.attention_implementation != "manual":
                raise ValueError(
                    "repeat_cache_window currently requires manual attention; "
                    "the SDPA/Flash path does not apply the repeat-cache mask."
                )
        
        # forward the GPT model itself
        if use_cache:
            idx, index_shift, cache_context = self.lm_cache(idx)
        else:
            index_shift = 0
            cache_context = None
        if getattr(self.transformer.wpe, "needs_iter", False):
            idx, pos_emb_closure = self.transformer.wpe(idx, iter=iter) # position embeddings of shape (1, t, n_embd)
        else:
            idx, pos_emb_closure = self.transformer.wpe(idx) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(x)
        x = pos_emb_closure.adapt_model_input(x, start_index=index_shift)

        intervention_x = None
        if has_intervention_depth:
            # This is the same canonical middle-stack input representation used
            # for the original row.  For the supported n_layer_begin == 0 case,
            # no learned begin stack has to be assigned an ambiguous role.
            intervention_x = self.transformer.wte(intervention_input_ids)
            intervention_x = self.transformer.drop(intervention_x)
            intervention_x = pos_emb_closure.adapt_model_input(
                intervention_x, start_index=index_shift
            )
       
        for block in self.transformer.h_begin:
            x = block(x, pos_emb_closure, cache_context, start_index=index_shift,)


        # Evaluation-only repeat diagnostics use these snapshots to determine
        # whether the shared middle stack converges, cycles, or leaves the
        # representation manifold. The first entry is the post-begin state.

        repeat_states = [x] if return_repeat_states else None
        repeat_logits=[] if return_repeat_logits else None
        if return_repeat_logits and len(self.transformer.h_end) != 0:
            raise ValueError(
                "Per-repeat decoding requires n_layer_end == 0."
            )

        B, T, D = x.shape
        attention_diagnostics = ([] if return_attention_diagnostics else None)
        # total_expected_length = self.n_repeat * T
        total_expected_length = (repeats + (recall_repeats if delayed_recall else 0) ) * T          # TODO done
        for block in self.transformer.h_mid:
            block.attn.init_cache(total_expected_length)
        recall_intervention_metadata = None

        if delayed_recall:
            recall_intervention_metadata = {
                "condition": (
                    recall_cache_intervention
                    if recall_cache_intervention is not None
                    else "baseline"
                ),
                "target_repeat": recall_target_repeat,
                "recall_age": recall_age,
                "num_evolution_repeats": repeats,
                "num_recall_repeats": recall_repeats,
                "target_cache_block_zero_based": recall_target_repeat - 1,
                "target_token_start": (
                    (recall_target_repeat - 1) * T
                ),
                "target_token_end_exclusive": (
                    recall_target_repeat * T
                ),
                "value_permutations": 0,
                "masked_attention_calls": 0,
                "layers": [],
            }
        
        try:
            for rep_idx in range(1, repeats + 1):
                relative_age = repeats - rep_idx + 1
                
                controller = make_controller(
                    ForwardBackwardEmbedding.FORWARD,
                    relative_age,
                )
    
                x = x + controller
    
                if (
                    intervention_x is not None
                    and rep_idx == intervention_source_depth + 1
                ):
                    # Replace only the current recurrent representation.  Each
                    # middle block retains the K/V entries accumulated during
                    # the free rollout through intervention_source_depth.
                    x = intervention_x
                active_repeat_cache_window = repeat_cache_window
                if has_cache_reset and rep_idx > cache_reset_source_depth:
                    # Permanently hide the pre-reset segment while allowing a
                    # new cache history to grow under the baseline window.
                    # Physical K/V storage and append positions remain intact.
                    post_reset_repeats = rep_idx - cache_reset_source_depth
                    active_repeat_cache_window = (
                        post_reset_repeats
                        if repeat_cache_window is None
                        else min(post_reset_repeats, repeat_cache_window)
                    )
                for mid_idx, block in enumerate(self.transformer.h_mid):
                    x, diagnostics = block(
                        x,
                        pos_emb_closure,
                        cache_context,
                        start_index=index_shift,
                        collect_attention_diagnostics=(
                            return_attention_diagnostics
                        ),
                        attention_diagnostics_recent_window=(
                            attention_diagnostics_recent_window
                        ),
                        repeat_cache_window=(active_repeat_cache_window),
                    )

                    if diagnostics is not None:
                        diagnostics["repeat_index"] = rep_idx
                        diagnostics["middle_layer_index"] = mid_idx
                        attention_diagnostics.append(diagnostics)


                if self.controller_application == "subtract":
                    x = x - controller

                if return_repeat_states:
                    repeat_states.append(x)

                if return_repeat_logits: # repeat loss and query loss are calculated seperately do not append query logits to repeat logits
                    decoded_state = self.transformer.ln_f(x)
                    decoded_logits = self.lm_head(decoded_state)
                    repeat_logits.append(decoded_logits)
            if return_repeat_logits:
                repeat_logits = torch.stack(repeat_logits, dim=1)

            if (delayed_recall and recall_cache_intervention== "target-value-corruption"):
                for middle_layer_index, block in enumerate(
                    self.transformer.h_mid
                ):
                    layer_metadata = (
                        block.attn.permute_cached_target_values(
                            target_repeat=recall_target_repeat,
                            tokens_per_repeat=T,
                            permutation_offset=(
                                recall_value_permutation_offset
                            ),
                        )
                    )
                    recall_intervention_metadata["layers"].append(
                        {
                            "middle_layer_index_zero_based": (
                                middle_layer_index
                            ),
                            **layer_metadata,
                        }
                    )
                    recall_intervention_metadata[
                        "value_permutations"
                    ] += 1
            if delayed_recall:
                controller = make_controller(
                    ForwardBackwardEmbedding.RECALL,
                    recall_age,
                )
    
                for recall_idx in range(1, recall_repeats+1):
                        
                    x = x + controller

                    for mid_idx,block in enumerate( self.transformer.h_mid):
                            
                        active_recall_mask = (
                            recall_cache_intervention
                            if recall_cache_intervention
                            in RECALL_ATTENTION_MASK_INTERVENTIONS
                            else None
                        )

                        x, diagnostics = block(
                            x,
                            pos_emb_closure,
                            cache_context,
                            start_index=index_shift,
                            collect_attention_diagnostics=(
                                return_attention_diagnostics
                            ),
                            attention_diagnostics_recent_window=(
                                attention_diagnostics_recent_window
                            ),
                            repeat_cache_window=active_repeat_cache_window,
                            recall_cache_intervention=active_recall_mask,
                            recall_target_repeat=recall_target_repeat,
                            evolution_repeats=repeats,
                            recall_index=recall_idx,
                        )

                        if active_recall_mask is not None:
                            recall_intervention_metadata[
                                "masked_attention_calls"
                            ] += 1
                        if diagnostics is not None:
                            diagnostics["repeat_index"] = repeats+recall_idx        # TODO
                            diagnostics["middle_layer_index"] = mid_idx  # TODO
                            attention_diagnostics.append(diagnostics)


                    if self.controller_application == "subtract":
                        x = x - controller

        finally:
            for block in self.transformer.h_mid:
                block.attn.drop_cache()

        for block in self.transformer.h_end:
            x = block(x, pos_emb_closure, cache_context, start_index=index_shift)
        
        x = self.transformer.ln_f(x)

        # if use_cache:
        #     x = self.lm_cache.get_final_logits(x)
        
        # if targets is not None:
        #     # if we are given some desired targets also calculate the loss
        #     logits = self.lm_head(x)
        #     cross_entropy_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        #     loss = cross_entropy_loss
        # else:
        #     # inference-time mini-optimization: only forward the lm_head on the very last position
        #     logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        #     loss = None
        #     cross_entropy_loss = None
        # logits = logits if get_logits else None
        # return {'logits': logits, 'loss': loss, 'cross_entropy_loss': cross_entropy_loss, 'average_depth': torch.as_tensor(repeats) * len(self.transformer.h_mid) + len(self.transformer.h_begin) + len(self.transformer.h_end)}
        if use_cache:
            x = self.lm_cache.get_final_logits(x)
        if targets is not None or return_all_logits:
            logits = self.lm_head(x)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            loss = None
        logits = logits if get_logits else None
        executed_repeats = repeats + (
            recall_repeats if delayed_recall else 0
        )

        average_depth = (
            executed_repeats * len(self.transformer.h_mid)
            + len(self.transformer.h_begin)
            + len(self.transformer.h_end)
        )
        result = {
            'logits': logits,
            'loss': loss,
            'average_depth': torch.as_tensor(average_depth, device=idx.device),
        }
        if return_repeat_states:
            result['repeat_states'] = repeat_states

        if return_attention_diagnostics:
            result['attention_diagnostics'] = attention_diagnostics

        if return_repeat_logits:
            result["repeat_logits"] = repeat_logits

        if recall_intervention_metadata is not None:
            result["recall_cache_intervention"] = (
                recall_intervention_metadata
            )
        return result
    def clear_state(self):
        self.lm_cache.clear_state()

    def crop_sequence_length(self, sequence_length):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:sequence_length,:sequence_length]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # TODO
        pass

    def get_parameter_group_specs(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        return [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = idx if idx.size(1) <= self.config.sequence_length else idx[:, -self.config.sequence_length:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, get_logits=True)['logits']
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, top_k=None):
        idx = torch.tensor(self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})).view(1,-1).to(self.lm_head.weight.device)
        out_idx = self.generate(idx, max_new_tokens, temperature, top_k).view(-1).to('cpu').numpy()
        return self.tokenizer.decode(out_idx)
