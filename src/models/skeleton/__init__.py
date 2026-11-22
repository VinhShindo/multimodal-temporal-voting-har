from .barlow_twins import BarlowTwinsSkeleton
from .ctrgcn import CTRGCN, CTRGCNPart
from .ms_g3d import MSG3D, build_msg3d
from .paper_architectures import SkeletonGraphClassifier
from .sta_gcn import STAGCN, build_sta_gcn
from .stgcn import STGCN, build_stgcn

__all__ = [
    "CTRGCN",
    "CTRGCNPart",
    "STGCN",
    "MSG3D",
    "STAGCN",
    "BarlowTwinsSkeleton",
    "SkeletonGraphClassifier",
    "build_model",
]


def build_model(arch: str = "ctrgcn_head", num_classes: int = 35, num_joints: int = 17,
               in_channels: int = None, target_len: int = 64, **kw):
    arch = arch.lower()
    if arch == "stgcn":
        return build_stgcn(num_classes=num_classes, num_joints=num_joints,
                           in_channels=in_channels or 6, dropout=kw.get("dropout", 0.3))
    if arch == "ms_g3d":
        return build_msg3d(num_classes=num_classes, num_joints=num_joints,
                           in_channels=in_channels or 6, dropout=kw.get("dropout", 0.3))
    if arch == "sta_gcn":
        return build_sta_gcn(num_classes=num_classes, num_joints=num_joints,
                             in_channels=in_channels or 6, dropout=kw.get("dropout", 0.3))
    if arch == "barlow_twins":
        return BarlowTwinsSkeleton(in_channels=in_channels or 6,
                                  num_classes=num_classes,
                                  target_len=target_len,
                                  dropout=kw.get("dropout", 0.3))
    if arch == "ctrgcn":
        return CTRGCN(num_classes, num_joints, in_channels or 6,
                      dropout=kw.get("dropout", 0.3))
    common = dict(dropout=kw.get("dropout", 0.4), depth=kw.get("tf_depth", 2),
                  heads=kw.get("tf_heads", 4), drop_path=kw.get("drop_path", 0.1),
                  part_drop=kw.get("part_drop", 0.1))
    if arch == "ctrgcn_head":
        return CTRGCNPart(num_classes, num_joints, in_channels or 11,
                          target_len, **common)
    raise ValueError(f"Unknown architecture: {arch}")
