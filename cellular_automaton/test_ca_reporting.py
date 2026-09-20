import json

from cellular_automaton.ca_reporting import (
    build_run_manifest,
    normalize_training_stats,
    validate_manifest,
    write_eval_metrics,
    write_run_manifest,
)


def sample_args(seed=1, data_seed=11):
    return {
        "dataset": "rule30",
        "model": "but_full_depth",
        "exp_name": "test",
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 32,
        "sequence_length": 64,
        "attention_mode": "bidirectional",
        "attention_implementation": "manual",
        "positional_encoder": "rotary",
        "n_repeat": 2,
        "vocab_size": 2,
        "opt": "adamw",
        "lr": 0.001,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "batch_size": 32,
        "acc_steps": 1,
        "iterations": 100,
        "ca_train_pairs": [[1, 1], [2, 2], [4, 4]],
        "ca_train_samples": 1000,
        "ca_train_num_cells": 64,
        "ca_data_mode": "materialized",
        "ca_num_workers": 4,
        "ca_eval_num_cells": [64],
        "ca_val_samples": 100,
        "ca_test_samples": 100,
        "ca_extrapolation_val_pairs": [[3, 3]],
        "ca_final_eval_pairs": [[3, 3], [5, 5]],
        "ca_best_metric": "exact_sequence_accuracy",
        "ca_delayed_best_metric": "cell_accuracy",
        "ca_recall_repeats": 2,
        "ca_best_length": 64,
        "ca_extrapolation_best_metric": "cell_accuracy",
        "seed": seed,
        "data_seed": data_seed,
        "ca_val_seed": 100,
        "ca_test_seed": 200,
    }


def sample_stats():
    metric = {
        "loss": 0.1,
        "cell_accuracy": 0.9,
        "exact_sequence_accuracy": 0.2,
        "position_accuracy": [0.8, 1.0],
    }
    diagnostics = {
        "64": {
            "repeat_horizon_matrix": {
                "repeats_3": {"steps_3": metric},
            },
            "best_matching_horizon": {
                "repeats_3": {
                    "by_cell_accuracy": 3,
                    "cell_accuracy": 0.9,
                }
            },
            "decoded_recurrence": {
                "repeats_3": {
                    "rule30_from_previous_decoded": metric,
                    "versus_previous_decoded": {
                        "cell_accuracy": 0.8,
                    },
                }
            },
            "decoded_similarity": {
                "repeats_2": {
                    "repeats_3": {"cosine_similarity": 0.7}
                }
            },
            "hidden_state_similarity": {
                "available": True,
                "similarity_matrix": {
                    "repeats_2": {
                        "repeats_3": {
                            "cosine_similarity": 0.6,
                            "normalized_mse": 0.4,
                        }
                    }
                },
            },
        }
    }
    clean_state_transitions = {
        "64": {
            "max_transition_depth": 3,
            "num_batches": 1,
            "transitions": {
                "steps_2_to_3": {
                    "source_ca_steps": 2,
                    "target_ca_steps": 3,
                    "num_repeats": 3,
                    "intervention_source_depth": 2,
                    "metrics": metric,
                }
            },
        }
    }
    retrieval = {
        "requested_repeat_is_best_rate": 0.75,
        "requested_repeat_is_unique_best_rate": 0.5,
        "requested_repeat_mean_rank": 1.25,
        "requested_repeat_mean_reciprocal_rank": 0.875,
        "requested_repeat_mean_margin_over_closest_wrong": 0.1,
        "requested_repeat_positive_margin_rate": 0.75,
        "total_sequences": 4,
    }
    internal_macro = {
        **retrieval,
        "decoded_requested_repeat_cell_accuracy": 0.8,
        "decoded_requested_repeat_exact_sequence_accuracy": 0.6,
    }
    collision = {
        "cell_agreement": 0.4,
        "exact_row_collision_rate": 0.1,
        "exact_row_collisions": 1,
        "total_sequences": 10,
    }
    delayed_query = {
        "num_repeats": 3,
        "query_repeat": 1,
        "recall_age": 2,
        "is_no_op": False,
        "metrics": metric,
        "ground_truth_comparison_by_repeat": {
            "repeat_1": metric,
            "repeat_2": {**metric, "matthews_correlation": 0.3},
            "repeat_3": {**metric, "matthews_correlation": 0.2},
        },
        "ground_truth_state_collision_by_repeat": {
            "repeat_1": {**collision, "exact_row_collision_rate": 1.0},
            "repeat_2": collision,
            "repeat_3": collision,
        },
        "ground_truth_retrieval": retrieval,
        "internal_consistency": {
            "decoded_requested_repeat": metric,
            "decoded_comparison_by_repeat": {
                "repeat_1": metric,
                "repeat_2": metric,
                "repeat_3": metric,
            },
            "decoded_state_collision_by_repeat": {
                "repeat_1": {**collision, "exact_row_collision_rate": 1.0},
                "repeat_2": collision,
                "repeat_3": collision,
            },
            "requested_repeat_logit_similarity": {
                "cosine_similarity": 0.9,
                "normalized_mse": 0.1,
            },
            "logit_similarity_by_repeat": {
                "repeat_1": {"cosine_similarity": 0.9, "normalized_mse": 0.1},
                "repeat_2": {"cosine_similarity": 0.5, "normalized_mse": 0.5},
                "repeat_3": {"cosine_similarity": 0.4, "normalized_mse": 0.6},
            },
            "decoded_retrieval": retrieval,
            "cosine_retrieval": retrieval,
        },
    }
    delayed_pair = {
        "ca_steps": 3,
        "num_repeats": 3,
        "queries": {"query_repeat_1": delayed_query},
        "all_queries_macro": metric,
        "nontrivial_queries_macro": metric,
        "all_queries_internal_consistency_macro": internal_macro,
        "nontrivial_internal_consistency_macro": internal_macro,
        "all_queries_ground_truth_retrieval_macro": retrieval,
        "nontrivial_ground_truth_retrieval_macro": retrieval,
    }
    delayed_summary = {
        "all_queries_pair_macro": metric,
        "all_internal_consistency_pair_macro": internal_macro,
        "all_ground_truth_retrieval_pair_macro": retrieval,
        "nontrivial_queries_pair_macro": metric,
        "nontrivial_internal_consistency_pair_macro": internal_macro,
        "nontrivial_ground_truth_retrieval_pair_macro": retrieval,
    }
    return {
        "eval": {
            "50": {
                "in_distribution": {
                    "steps_1_repeats_1": {
                        "ca_steps": 1,
                        "num_repeats": 1,
                        "by_length": {"64": metric},
                    }
                },
                "extrapolation_validation": {
                    "steps_3_repeats_3": {
                        "ca_steps": 3,
                        "num_repeats": 3,
                        "by_length": {"64": metric},
                    }
                },
                "repeat_diagnostics": diagnostics,
                "delayed_recall": {
                    "steps_3_repeats_3": delayed_pair,
                },
                "delayed_recall_summary": delayed_summary,
                "training_exposure": {
                    "accounting_exact": True,
                    "legacy_approximated_steps": 0,
                    "materialized_training_rows": 1000,
                    "total_optimizer_steps": 50,
                    "total_microbatches": 50,
                    "total_examples_seen": 6400,
                    "total_cells_seen": 409600,
                    "equivalent_dataset_passes": 6.4,
                    "by_mode": {
                        "normal": {
                            "optimizer_steps": 25,
                            "microbatches": 25,
                            "examples_seen": 3200,
                            "cells_seen": 204800,
                            "example_fraction": 0.5,
                            "optimizer_step_fraction": 0.5,
                        },
                        "delayed": {
                            "optimizer_steps": 25,
                            "microbatches": 25,
                            "examples_seen": 3200,
                            "cells_seen": 204800,
                            "example_fraction": 0.5,
                            "optimizer_step_fraction": 0.5,
                        },
                    },
                    "by_training_pair": {
                        "steps_1_repeats_1": {
                            "ca_steps": 1,
                            "num_repeats": 1,
                            "optimizer_steps": 25,
                            "microbatches": 25,
                            "examples_seen": 3200,
                            "cells_seen": 204800,
                            "example_fraction": 0.5,
                            "optimizer_step_fraction": 0.5,
                            "by_mode": {
                                "normal": {
                                    "optimizer_steps": 12,
                                    "examples_seen": 1536,
                                },
                                "delayed": {
                                    "optimizer_steps": 13,
                                    "examples_seen": 1664,
                                },
                            },
                            "delayed_by_query_repeat": {
                                "query_repeat_1": {
                                    "query_repeat": 1,
                                    "recall_age": 0,
                                    "target_steps": 1,
                                    "optimizer_steps": 13,
                                    "examples_seen": 1664,
                                }
                            },
                        }
                    },
                },
            }
        },
        "checkpoint_analysis": {
            "best_id": {
                "checkpoint": {"step": 50},
                "task_metrics": {
                    "in_distribution": {
                        "training_pairs": [[1, 1]],
                        "by_pair": {
                            "steps_1_repeats_1": {
                                "ca_steps": 1,
                                "num_repeats": 1,
                                "by_length": {"64": metric},
                            }
                        },
                    },
                    "internal_repeat_extrapolation": {
                        "steps_3_repeats_3": {
                            "ca_steps": 3,
                            "num_repeats": 3,
                            "by_length": {"64": metric},
                        }
                    },
                    "external_rollout": {
                        "steps_8": {
                            "ca_steps": 8,
                            "model_calls": 4,
                            "num_repeats_per_call": 2,
                            "by_length": {"64": metric},
                        }
                    },
                },
                "repeat_diagnostics": diagnostics,
                "preserved_cache_clean_state_transitions": (
                    clean_state_transitions
                ),
            },
            "best_extrapolation_strict": None,
            "best_extrapolation_unconstrained": None,
            "best_delayed_recall": {
                "checkpoint": {"step": 50, "length": 64},
                "task_metrics": {},
                "delayed_recall": {
                    "steps_3_repeats_3": delayed_pair,
                },
                "delayed_recall_summary": delayed_summary,
            },
            "best_internal_recall": {
                "checkpoint": {"step": 50, "length": 64},
                "task_metrics": {},
                "delayed_recall": {
                    "steps_3_repeats_3": delayed_pair,
                },
                "delayed_recall_summary": delayed_summary,
            },
        },
    }


def test_manifest_preserves_arbitrary_training_pairs(tmp_path):
    manifest = build_run_manifest(
        sample_args(), tmp_path / "run_1__but", tmp_path / "checkpoints"
    )
    assert validate_manifest(manifest) == []
    assert manifest["training"]["pairs"] == [
        {"ca_steps": 1, "num_repeats": 1},
        {"ca_steps": 2, "num_repeats": 2},
        {"ca_steps": 4, "num_repeats": 4},
    ]
    assert manifest["model"]["attention_mode"] == "bidirectional"
    assert manifest["model"]["attention_implementation"] == "manual"
    assert manifest["forward_policy"] == {
        "repeat_cache_policy": "full",
        "repeat_cache_window": None,
        "forward_policy_label": "cache_full",
    }
    assert manifest["seeds"]["seed"] == 1
    assert manifest["seeds"]["data_seed"] == 11
    assert manifest["training"]["ca_recall_repeats"] == 2
    assert manifest["checkpoint_selection"]["ca_delayed_best_metric"] == (
        "cell_accuracy"
    )


def test_manifest_records_recent_cache_policy(tmp_path):
    args = sample_args()
    args.update(
        {
            "model": "ca_cotf_cache_attn",
            "repeat_cache_window": 4,
        }
    )

    manifest = build_run_manifest(
        args, tmp_path / "run_1__cotformer", tmp_path / "checkpoints"
    )

    assert manifest["forward_policy"] == {
        "repeat_cache_policy": "recent",
        "repeat_cache_window": 4,
        "forward_policy_label": "cache_recent_4",
    }


def test_normalization_exports_scalar_long_form_records(tmp_path):
    run_dir = tmp_path / "run_1__but"
    manifest = build_run_manifest(
        sample_args(), run_dir, tmp_path / "checkpoints"
    )
    records = normalize_training_stats(sample_stats(), manifest)

    assert all(record["repeat_cache_policy"] == "full" for record in records)
    assert all(record["repeat_cache_window"] is None for record in records)
    assert all(
        record["forward_policy_label"] == "cache_full" for record in records
    )

    roles = {record["evaluation_role"] for record in records}
    assert "extrapolation_validation" in roles
    assert "decoded_recurrence" in roles
    assert "preserved_cache_clean_state_transition" in roles
    assert "hidden_state_similarity" in roles
    assert "internal_repeat_extrapolation" in roles
    assert "external_rollout" in roles
    assert "training_exposure" in roles
    assert "training_exposure_by_pair" in roles
    assert "training_exposure_normal" in roles
    assert "training_exposure_by_pair_delayed" in roles
    assert "training_exposure_delayed_query" in roles
    assert "delayed_recall_ground_truth_candidate" in roles
    assert "delayed_recall_internal_decoded_candidate" in roles
    assert "delayed_recall_cosine_retrieval" in roles
    assert "delayed_recall_ground_truth_collision" in roles
    assert "delayed_recall_nontrivial_pair_macro" in roles
    assert "delayed_recall_all_queries_pair_macro" in roles
    assert "delayed_recall_all_internal_pair_macro" in roles
    assert not any(record["metric"] == "position_accuracy" for record in records)

    external = next(
        record
        for record in records
        if record["evaluation_role"] == "external_rollout"
        and record["metric"] == "cell_accuracy"
    )
    assert external["ca_steps"] == 8
    assert external["num_repeats"] == 2
    assert external["checkpoint_type"] == "best_id"
    assert external["data_split"] == "final_test"

    clean_transition = next(
        record
        for record in records
        if record["evaluation_role"]
        == "preserved_cache_clean_state_transition"
        and record["metric"] == "cell_accuracy"
    )
    assert clean_transition["ca_steps"] == 3
    assert clean_transition["num_repeats"] == 3
    assert clean_transition["repeat_from"] == 2
    assert clean_transition["repeat_to"] == 3
    assert clean_transition["checkpoint_type"] == "best_id"
    assert clean_transition["data_split"] == "final_test"

    exposure = next(
        record
        for record in records
        if record["evaluation_role"] == "training_exposure_by_pair"
        and record["metric"] == "examples_seen"
    )
    assert exposure["data_split"] == "training"
    assert exposure["ca_steps"] == 1
    assert exposure["num_repeats"] == 1
    assert exposure["value"] == 3200

    candidate_mcc = next(
        record
        for record in records
        if record["evaluation_role"]
        == "delayed_recall_ground_truth_candidate"
        and record["metric"] == "matthews_correlation"
        and record["repeat_from"] == 1
        and record["repeat_to"] == 2
        and record["data_split"] == "validation"
    )
    assert candidate_mcc["num_repeats"] == 3
    assert candidate_mcc["length"] == 64
    assert candidate_mcc["value"] == 0.3

    final_cosine_rank = next(
        record
        for record in records
        if record["evaluation_role"] == "delayed_recall_cosine_retrieval"
        and record["metric"] == "requested_repeat_mean_rank"
        and record["checkpoint_type"] == "best_delayed_recall"
    )
    assert final_cosine_rank["data_split"] == "final_test"
    assert final_cosine_rank["value"] == 1.25

    final_internal_cell = next(
        record
        for record in records
        if record["evaluation_role"] == "delayed_recall_all_internal_pair_macro"
        and record["metric"] == "decoded_requested_repeat_cell_accuracy"
        and record["checkpoint_type"] == "best_internal_recall"
    )
    assert final_internal_cell["data_split"] == "final_test"
    assert final_internal_cell["value"] == 0.8


def test_normalization_infers_policy_for_historical_manifest(tmp_path):
    args = sample_args()
    args["repeat_cache_window"] = 4
    manifest = build_run_manifest(
        args, tmp_path / "run_1__but", tmp_path / "checkpoints"
    )
    manifest.pop("forward_policy")

    records = normalize_training_stats(sample_stats(), manifest)

    assert records
    assert all(record["repeat_cache_policy"] == "recent" for record in records)
    assert all(record["repeat_cache_window"] == 4 for record in records)
    assert all(
        record["forward_policy_label"] == "cache_recent_4"
        for record in records
    )


def test_metrics_file_is_rewritten_without_resume_duplicates(tmp_path):
    run_dir = tmp_path / "run_1__but"
    manifest = build_run_manifest(
        sample_args(), run_dir, tmp_path / "checkpoints"
    )
    write_run_manifest(run_dir, manifest)
    path = write_eval_metrics(run_dir, sample_stats())
    first = path.read_text(encoding="utf-8")
    write_eval_metrics(run_dir, sample_stats())
    second = path.read_text(encoding="utf-8")

    assert first == second
    assert all(json.loads(line)["run_id"] == run_dir.name for line in second.splitlines())
