#!/usr/bin/env python3
"""Load a checkpoint, construct model+wrapper, run evaluation and print NLL/ECE."""
import argparse
from pathlib import Path
import torch
import logging

from untangle.utils.parsing import parser as default_parser
from untangle.utils import (
    resolve_data_config,
    create_model,
    wrap_model,
    setup_logging,
)
from train import setup_devices, setup_amp, create_loaders
from untangle.utils.model import load_checkpoint
from validate import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--method", type=str, required=True)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--extra-args", nargs=argparse.REMAINDER, default=None)
    args_ns = p.parse_args()

    # Use existing CLI defaults then override (parse empty argv to get defaults)
    default_args = default_parser.parse_args([])

    # Override fields
    default_args.method_name = args_ns.method
    default_args.initial_checkpoint_path = args_ns.checkpoint
    if args_ns.num_classes is not None:
        default_args.num_classes = args_ns.num_classes
    default_args.dataset = "hard/pathmnist"
    default_args.dataset_id = "hard/pathmnist"
    default_args.data_dir = Path("/tmp/pathmnist_real")
    default_args.data_dir_id = Path("/tmp/pathmnist_real")
    default_args.dataset_download = True
    default_args.evaluate_on_test_sets = True
    default_args.discard_ood_test_sets = True
    default_args.num_workers = 2
    default_args.num_eval_workers = 2
    default_args.storage_device = "cuda"
    default_args.batch_size = 128
    default_args.img_size = 224

    # Apply any extra args by naive parsing of known flags
    if args_ns.extra_args:
        it = iter(args_ns.extra_args)
        for tok in it:
            if tok == "--num-classes":
                default_args.num_classes = int(next(it))
            else:
                # set attribute if exists and next token not starting with --
                name = tok.lstrip("-" ).replace("-","_")
                try:
                    val = next(it)
                except StopIteration:
                    val = True
                # best effort cast
                if hasattr(default_args, name):
                    try:
                        current = getattr(default_args, name)
                        if isinstance(current, bool):
                            setattr(default_args, name, True)
                        elif isinstance(current, int):
                            setattr(default_args, name, int(val))
                        elif isinstance(current, float):
                            setattr(default_args, name, float(val))
                        else:
                            setattr(default_args, name, val)
                    except Exception:
                        setattr(default_args, name, val)

    # Ensure compatibility aliases from parse_args()
    if not hasattr(default_args, "soft_imagenet_label_dir"):
        default_args.soft_imagenet_label_dir = getattr(default_args, "soft_label_root", Path())

    setup_logging(default_args)
    device, storage_device = setup_devices(default_args)

    data_config = resolve_data_config(vars(default_args))
    amp_autocast, _ = setup_amp(device, default_args)

    model = create_model(
        model_name=default_args.model_name,
        pretrained=default_args.pretrained,
        num_classes=default_args.num_classes,
        in_chans=data_config["input_size"][0],
        model_kwargs=default_args.model_kwargs,
    )

    model = wrap_model(
        model=model,
        model_wrapper_name=default_args.method_name,
        reset_classifier=default_args.reset_classifier,
        weight_paths=default_args.weight_paths,
        num_hidden_features=default_args.num_hidden_features,
        mlp_depth=default_args.mlp_depth,
        stopgrad=default_args.stopgrad,
        num_hooks=default_args.num_hooks,
        module_type=default_args.module_type,
        module_name_regex=default_args.module_name_regex,
        dropout_probability=default_args.dropout_probability,
        use_filterwise_dropout=default_args.use_filterwise_dropout,
        num_mc_samples=default_args.num_mc_samples,
        num_mc_samples_integral=default_args.num_mc_samples_integral,
        num_mc_samples_cv=default_args.num_mc_samples_cv,
        rbf_length_scale=default_args.rbf_length_scale,
        ema_momentum=default_args.ema_momentum,
        matrix_rank=default_args.matrix_rank,
        use_het=default_args.use_het,
        temperature=default_args.temperature,
        pred_type=default_args.pred_type,
        hessian_structure=default_args.hessian_structure,
        use_low_rank_cov=default_args.use_low_rank_cov,
        max_rank=default_args.max_rank,
        magnitude=default_args.magnitude,
        num_heads=default_args.num_heads,
        likelihood=default_args.likelihood,
        use_spectral_normalization=default_args.use_spectral_normalization,
        spectral_normalization_iteration=default_args.spectral_normalization_iteration,
        spectral_normalization_bound=default_args.spectral_normalization_bound,
        use_spectral_normalized_batch_norm=default_args.use_spectral_normalized_batch_norm,
        use_tight_norm_for_pointwise_convs=default_args.use_tight_norm_for_pointwise_convs,
        num_random_features=default_args.num_random_features,
        gp_kernel_scale=default_args.gp_kernel_scale,
        gp_output_bias=default_args.gp_output_bias,
        gp_random_feature_type=default_args.gp_random_feature_type,
        use_input_normalized_gp=default_args.use_input_normalized_gp,
        gp_cov_momentum=default_args.gp_cov_momentum,
        gp_cov_ridge_penalty=default_args.gp_cov_ridge_penalty,
        gp_input_dim=default_args.gp_input_dim,
        latent_dim=default_args.latent_dim,
        num_density_components=default_args.num_density_components,
        use_batched_flow=default_args.use_batched_flow,
        edl_activation=default_args.edl_activation,
        checkpoint_path=default_args.initial_checkpoint_path,
    )

    model.to(device=device)

    # Load the checkpoint into model
    load_checkpoint(model, args_ns.checkpoint)

    # Create loaders
    (
        train_loader,
        id_eval_loader,
        hard_id_eval_loader,
        id_test_loader,
        ood_uniform_test_loaders,
        ood_varied_test_loaders,
        varied_s2_eval_loader,
    ) = create_loaders(default_args, data_config, device)

    # Run evaluate on id_test_loader to get metrics
    metrics = evaluate(
        model=model,
        loader=id_test_loader,
        loader_name=default_args.dataset_id.replace("/","_"),
        device=device,
        storage_device=storage_device,
        amp_autocast=amp_autocast,
        key_prefix="id_test",
        output_dir=Path("results/pathmnist_uncertainty_decomposition"),
        is_upstream_dataset=True,
        is_test_dataset=True,
        is_soft_dataset=("soft" in default_args.dataset_id),
        args=default_args,
    )

    # Print selected metrics: NLL (log_prob_score_hard_bma_aleatoric) and ECE
    nll_key = "id_test_log_prob_score_hard_bma_aleatoric"
    nll_key_orig = "id_test_log_prob_score_hard_bma_aleatoric_original"
    ece_key = "id_test_one_minus_max_probs_of_bma_ece_hard_bma_correctness_original"  # depends on estimator

    # Find any NLL keys in metrics
    print(f"Metrics for checkpoint: {args_ns.checkpoint}")
    for k in sorted(metrics.keys()):
        if "log_prob_score" in k or "ece_" in k or k.startswith("id_test_log_prob") or k.startswith("id_test_ece"):
            print(f"{k}: {metrics[k]}")

    # Also print a compact selection
    for key in [nll_key_orig, nll_key, "id_test_one_minus_max_probs_of_bma_ece_hard_bma_correctness_original"]:
        if key in metrics:
            print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
