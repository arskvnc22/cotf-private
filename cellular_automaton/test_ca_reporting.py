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
                "training_exposure": {
                    "accounting_exact": True,
                    "legacy_approximated_steps": 0,
                    "materialized_training_rows": 1000,
                    "total_optimizer_steps": 50,
                    "total_microbatches": 50,
                    "total_examples_seen": 6400,
                    "total_cells_seen": 409600,
                    "equivalent_dataset_passes": 6.4,
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
