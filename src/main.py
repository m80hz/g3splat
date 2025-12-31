import os
from pathlib import Path

# # ---- FORCE scratch & caches to match Slurm script ----
# USER = os.environ.get("USER", "unknown")
# JOB  = os.environ.get("SLURM_JOB_ID", "manual")
# # set this to the same base in the Slurm script.
# BASE = os.environ.get("SCRATCH_BASE", f"/scratchdata1/users/{USER}")

# FORCED_TMPDIR = f"{BASE}/tmp-{JOB}"
# FORCED_WANDB  = f"{FORCED_TMPDIR}/wandb_cache"
# FORCED_CUDA   = f"{FORCED_TMPDIR}/cuda"

# os.environ["TMPDIR"] = FORCED_TMPDIR
# os.environ["TMP"] = FORCED_TMPDIR
# os.environ["TEMP"] = FORCED_TMPDIR
# os.environ["WANDB_CACHE_DIR"] = FORCED_WANDB
# os.environ["CUDA_CACHE_PATH"] = FORCED_CUDA

# # ensure they exist
# for k in ("TMPDIR", "WANDB_CACHE_DIR", "CUDA_CACHE_PATH"):
#     Path(os.environ[k]).mkdir(parents=True, exist_ok=True)

# print("[main.py] (forced) TMPDIR        =", os.environ["TMPDIR"])
# print("[main.py] (forced) WANDB_CACHE_DIR=", os.environ["WANDB_CACHE_DIR"])
# print("[main.py] (forced) CUDA_CACHE_PATH=", os.environ["CUDA_CACHE_PATH"])
# # ---------------------------------------

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from src.misc.weight_modify import checkpoint_filter_fn
from src.model.distiller import get_distiller

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


# Determine global rank (default to 0 if not set)
global_rank = int(os.environ.get("LOCAL_RANK", 0))


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled" and global_rank == 0:
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    # Only add ModelCheckpoint if this is rank 0
    if global_rank == 0:
        callbacks.append(
            ModelCheckpoint(
                output_dir / "checkpoints",
                every_n_train_steps=cfg.checkpointing.every_n_train_steps,
                save_top_k=cfg.checkpointing.save_top_k,
                save_weights_only=cfg.checkpointing.save_weights_only,
                save_last=True,
                monitor="info/global_step",
                mode="max",
            )
        )
        callbacks[-1].CHECKPOINT_EQUALS_CHAR = '_'

    # callbacks.append(PrintGpuUsageCallback())

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy="ddp_find_unused_parameters_true",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        inference_mode=False if (cfg.mode == "test" and cfg.test.align_pose) else True,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    distiller = None
    if cfg.train.distiller:
        distiller = get_distiller(cfg.train.distiller)
        distiller = distiller.eval()

    # Load the encoder weights.
    if cfg.model.encoder.pretrained_weights and cfg.mode == "train":
        weight_path = cfg.model.encoder.pretrained_weights
        ckpt_weights = torch.load(weight_path, map_location='cpu')
        if 'model' in ckpt_weights:
            ckpt_weights = ckpt_weights['model']
            ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        elif 'state_dict' in ckpt_weights:
            ckpt_weights = ckpt_weights['state_dict']
            ckpt_weights = {k[8:]: v for k, v in ckpt_weights.items() if k.startswith('encoder.')}
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        else:
            raise ValueError(f"Invalid checkpoint format: {weight_path}")

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder),
        get_losses(cfg.loss),
        step_tracker,
        distiller=distiller,
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    train()
