from .loss import Loss
from .loss_orient import LossOrient, LossOrientCfgWrapper
from .loss_align import LossAlign, LossAlignCfgWrapper
from .loss_opacity import LossOpacity, LossOpacityCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper

LOSSES = {
    LossOrientCfgWrapper: LossOrient,
    LossAlignCfgWrapper: LossAlign,
    LossOpacityCfgWrapper: LossOpacity,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
}

LossCfgWrapper = LossLpipsCfgWrapper | LossMseCfgWrapper | LossOrientCfgWrapper | LossAlignCfgWrapper | LossOpacityCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
