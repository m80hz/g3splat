import json
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from lightning import Trainer

from src.evaluation.mesh_evaluator import ScanNetMeshEvaluator as MeshEvaluator
from src.loss import get_losses, LossCfgWrapper
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder


# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import (
        load_typed_config,
        ModelCfg,
        CheckpointingCfg,
        separate_loss_cfg_wrappers,
        separate_dataset_cfg_wrappers,
    )
    from src.dataset.data_module import DataLoaderCfg, DataModule, DatasetCfgWrapper
    from src.evaluation.evaluation_cfg import EvaluationCfg
    from src.global_cfg import set_cfg


@dataclass
class RootCfg:
    evaluation: EvaluationCfg
    dataset: list[DatasetCfgWrapper]
    data_loader: DataLoaderCfg
    model: ModelCfg
    checkpointing: CheckpointingCfg
    loss: list[LossCfgWrapper]
    seed: int


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def evaluate(cfg_dict: DictConfig):
    cfg = load_typed_config(
        cfg_dict,
        RootCfg,
        {
            list[LossCfgWrapper]: separate_loss_cfg_wrappers,
            list[DatasetCfgWrapper]: separate_dataset_cfg_wrappers,
        },
    )
    set_cfg(cfg_dict)
    torch.manual_seed(cfg.seed)

    # Mesh-eval-only override: ensure baseline normalization for ScanNet depth datasets
    # regardless of the default used in other eval scripts.
    for i, ds in enumerate(cfg.dataset):
        # Each item is a typed wrapper. Detect the ScanNet depth wrapper by attribute.
        if hasattr(ds, "scannet_depth"):
            ds.scannet_depth.make_baseline_1 = True
    # Also mirror this into the underlying raw cfg_dict so any consumers reading it directly see the override.
    try:
        if "dataset" in cfg_dict:
            # When composed via group, dataset entries are keyed by their group name.
            if "scannet_depth" in cfg_dict["dataset"]:
                cfg_dict["dataset"]["scannet_depth"]["make_baseline_1"] = True
            # Fallback for grouped composition under _group_
            if "_group_" in cfg_dict and "scannet_depth" in cfg_dict["_group_"]:
                cfg_dict["_group_"]["scannet_depth"]["make_baseline_1"] = True
    except Exception:
        pass

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    ckpt_weights = torch.load(cfg.checkpointing.load, map_location="cpu")["state_dict"]
    # remove the prefix "encoder.", need to judge if is at start of key
    ckpt_weights = {k[8:] if k.startswith("encoder.") else k: v for k, v in ckpt_weights.items()}
    missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=True)

    trainer = Trainer(max_epochs=-1, accelerator="gpu", inference_mode=False)
    mesh_evaluator = MeshEvaluator(
        cfg.evaluation,
        encoder,
        get_decoder(cfg.model.decoder),
        get_losses(cfg.loss),
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
    )

    metrics = trainer.test(mesh_evaluator, datamodule=data_module)

    cfg.evaluation.output_metrics_path.parent.mkdir(exist_ok=True, parents=True)
    with cfg.evaluation.output_metrics_path.open("w") as f:
        json.dump(metrics[0] if len(metrics) > 0 else {}, f)


if __name__ == "__main__":
    evaluate()
