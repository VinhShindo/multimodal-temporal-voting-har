"""Architecture provenance used by benchmark reports.

The registry deliberately distinguishes a paper architecture from a project
variant. Matching a paper's broad family is not the same as reproducing its
operator, depth, training objective, and input protocol.
"""

ARCHITECTURE_REGISTRY = {
    "skeleton": {
        "stgcn": {
            "status": "paper_family_adapted",
            "note": "ST-GCN operator and 10-block schedule; COCO-17 graph and MMAct head are adapted.",
            "reference": "Yan et al. 2018",
        },
        "ctrgcn": {
            "status": "paper_family_adapted",
            "note": "CTR-GC backbone adapted to COCO-17 and project streams; not an official checkpoint reproduction.",
            "reference": "Liu et al. 2021",
        },
        "ms_g3d": {
            "status": "custom_inspired",
            "note": "Simplified multi-scale graph/temporal block; it is not the paper's G3D + MS-G3D implementation.",
            "reference": "Liu et al. 2020",
        },
        "sta_gcn": {
            "status": "custom_inspired",
            "note": "Channel/spatial attention variant; it is not a faithful reproduction of the cited STA-GCN paper.",
            "reference": "Shiraki et al. 2020",
        },
        "barlow_twins": {
            "status": "hybrid_ssl_ablation",
            "note": "Supervised CE plus Barlow-Twins auxiliary loss; not the paper's standalone self-supervised protocol.",
            "reference": "Zbontar et al. 2021",
        },
        "ctrgcn_head": {
            "status": "custom_ablation",
            "note": "CTR-GCN plus project-specific part-token Transformer head.",
            "reference": "Liu et al. 2021 (backbone only)",
        },
        "dual": {
            "status": "custom_ablation",
            "note": "Dual global/upper-body CTR-GCN with project-specific head.",
            "reference": "Liu et al. 2021 (backbone only)",
        },
    },
    "imu": {
        "cnn_bilstm": {"status": "custom_baseline", "note": "Project CNN-BiLSTM baseline.", "reference": None},
        "cnn_bilstm_cbam": {"status": "custom_variant", "note": "1D adaptation of CBAM around CNN-BiLSTM.", "reference": "Woo et al. 2018 (CBAM)"},
        "tcn": {"status": "paper_family_adapted", "note": "Residual dilated TCN variant; exact channel/depth schedule is project-specific.", "reference": "Bai et al. 2018"},
        "transformer": {"status": "custom_baseline", "note": "Compact temporal Transformer with project-specific convolutional stem.", "reference": "Vaswani et al. 2017 (operator only)"},
        "xtinyhar": {"status": "custom_approximation", "note": "Patch Transformer approximation; the published KD/XAI training protocol is not implemented.", "reference": "XTinyHAR 2025"},
        "inceptiontime": {"status": "custom_approximation", "note": "InceptionTime-inspired blocks; paper ensemble and exact schedule are not reproduced.", "reference": "Fawaz et al. 2020"},
        "fckd": {"status": "custom_variant", "note": "Project FCKD-style compact student; verify against the intended FCKD paper before publication.", "reference": None},
    },
    "rgb": {
        "r2plus1d_18": {"status": "official_backbone_adapted", "note": "Torchvision R(2+1)D-18 with project classifier head.", "reference": "Tran et al. 2018"},
        "r3d_18": {"status": "official_backbone_adapted", "note": "Torchvision 3D ResNet-18 with project classifier head.", "reference": "Hara et al. 2018"},
        "mc3_18": {"status": "official_backbone_adapted", "note": "Torchvision MC3-18 with project classifier head.", "reference": "Medical/Video ResNet family"},
        "s3d": {"status": "official_backbone_adapted", "note": "Torchvision S3D Kinetics backbone with project classifier head.", "reference": "Xie et al. 2018"},
        "mvit_v2_s": {"status": "official_backbone_adapted", "note": "Torchvision MViTv2-S Kinetics backbone with project classifier head.", "reference": "Fang et al. 2022"},
        "videomae": {"status": "official_encoder_adapted", "note": "VideoMAE encoder/checkpoint with project classifier and preprocessing adapter.", "reference": "Tong et al. 2022"},
        "videomae2": {"status": "official_encoder_adapted", "note": "VideoMAE v2 student encoder mapping with project classifier; not full official pretraining reproduction.", "reference": "Wang et al. 2023 (VideoMAE v2)"},
    },
}


def architecture_metadata(modality: str, architecture: str) -> dict:
    """Return stable provenance fields for a benchmark row."""
    item = ARCHITECTURE_REGISTRY.get(modality, {}).get(architecture)
    if item is None:
        return {
            "architecture_status": "unknown",
            "architecture_note": "No provenance entry.",
            "paper_reference": None,
        }
    return {
        "architecture_status": item["status"],
        "architecture_note": item["note"],
        "paper_reference": item["reference"],
    }
