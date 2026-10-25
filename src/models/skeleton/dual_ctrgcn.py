"""Dual-branch CTR-GCN variant isolated in its own file."""

from src.models.skeleton.ctrgcn import DualBranchCTRGCN


__all__ = ["DualBranchCTRGCN"]


def build_dual_ctrgcn(num_classes: int = 35, num_joints: int = 17,
                     in_channels: int = 18, target_len: int = 64, **kwargs):
    return DualBranchCTRGCN(num_classes=num_classes,
                           num_joints=num_joints,
                           in_channels=in_channels,
                           target_len=target_len,
                           dropout=kwargs.get("dropout", 0.4),
                           depth=kwargs.get("tf_depth", 2),
                           heads=kwargs.get("tf_heads", 4),
                           drop_path=kwargs.get("drop_path", 0.1),
                           part_drop=kwargs.get("part_drop", 0.1))
