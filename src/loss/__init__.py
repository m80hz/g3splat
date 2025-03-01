from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_normal import LossNormal, LossNormalCfgWrapper
from .loss_grid import LossGrid, LossGridCfgWrapper
from .loss_opacity import LossOpacity, LossOpacityCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossNormalCfgWrapper: LossNormal,
    LossGridCfgWrapper: LossGrid,
    LossOpacityCfgWrapper: LossOpacity,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper | LossNormalCfgWrapper | LossGridCfgWrapper | LossOpacityCfgWrapper 


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
