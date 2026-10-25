"""VideoMAE V2 distilled ViT-small classifier.

Uses OpenGVLab's official K710-distilled student checkpoint and the matching
VideoMAE encoder implementation from Hugging Face Transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


CHECKPOINT_REPO = "OpenGVLab/VideoMAE2"
CHECKPOINT_FILE = "distill/vit_s_k710_dl_from_giant.pth"
NUM_FRAMES = 16


def _load_official_encoder_weights(model: nn.Module) -> None:
    from huggingface_hub import hf_hub_download

    checkpoint_path = hf_hub_download(
        repo_id=CHECKPOINT_REPO,
        filename=CHECKPOINT_FILE,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source = checkpoint.get("module", checkpoint.get("model", checkpoint))
    target = model.state_dict()
    converted = {}

    converted["videomae.embeddings.patch_embeddings.projection.weight"] = \
        source["patch_embed.proj.weight"]
    converted["videomae.embeddings.patch_embeddings.projection.bias"] = \
        source["patch_embed.proj.bias"]
    converted["fc_norm.weight"] = source["fc_norm.weight"]
    converted["fc_norm.bias"] = source["fc_norm.bias"]

    for index in range(12):
        source_prefix = f"blocks.{index}."
        target_prefix = f"videomae.encoder.layer.{index}."
        for source_name, target_name in (
            ("norm1.weight", "layernorm_before.weight"),
            ("norm1.bias", "layernorm_before.bias"),
            ("norm2.weight", "layernorm_after.weight"),
            ("norm2.bias", "layernorm_after.bias"),
            ("attn.proj.weight", "attention.output.dense.weight"),
            ("attn.proj.bias", "attention.output.dense.bias"),
            ("mlp.fc1.weight", "intermediate.dense.weight"),
            ("mlp.fc1.bias", "intermediate.dense.bias"),
            ("mlp.fc2.weight", "output.dense.weight"),
            ("mlp.fc2.bias", "output.dense.bias"),
        ):
            converted[target_prefix + target_name] = source[source_prefix + source_name]

        q_weights, k_weights, v_weights = source[source_prefix + "attn.qkv.weight"].chunk(3)
        converted[target_prefix + "attention.attention.query.weight"] = q_weights
        converted[target_prefix + "attention.attention.key.weight"] = k_weights
        converted[target_prefix + "attention.attention.value.weight"] = v_weights
        converted[target_prefix + "attention.attention.query.bias"] = \
            source[source_prefix + "attn.q_bias"]
        converted[target_prefix + "attention.attention.key.bias"] = torch.zeros_like(
            source[source_prefix + "attn.q_bias"])
        converted[target_prefix + "attention.attention.value.bias"] = \
            source[source_prefix + "attn.v_bias"]

    missing_keys = set(converted) - set(target)
    if missing_keys:
        raise RuntimeError(f"Unexpected VideoMAE2 mapping targets: {sorted(missing_keys)}")
    incompatible = [
        key for key, value in converted.items()
        if target[key].shape != value.shape
    ]
    if incompatible:
        raise RuntimeError(f"VideoMAE2 checkpoint shape mismatch: {incompatible}")
    model.load_state_dict(converted, strict=False)


class VideoMAE2Classifier(nn.Module):
    def __init__(self, num_classes: int = 35, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        try:
            from transformers import VideoMAEConfig, VideoMAEForVideoClassification
        except ImportError as error:
            raise ImportError(
                "VideoMAE2 requires transformers; install project requirements."
            ) from error

        config = VideoMAEConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            num_frames=NUM_FRAMES,
            tubelet_size=2,
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            intermediate_size=1536,
            num_labels=num_classes,
            use_mean_pooling=True,
            layer_norm_eps=1e-6,
        )
        self.model = VideoMAEForVideoClassification(config)
        if pretrained:
            _load_official_encoder_weights(self.model)
        classifier = self.model.classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier.in_features, num_classes),
        )
        self.register_buffer(
            "source_mean", torch.tensor([0.45, 0.45, 0.45]).view(1, 1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            "source_std", torch.tensor([0.225, 0.225, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            "model_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            "model_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False)

    def forward(self, x):
        # Dataset input is (B, T, C, H, W); VideoMAE uses that same layout.
        if x.shape[1] < NUM_FRAMES:
            x = x.permute(0, 2, 1, 3, 4)
            x = F.interpolate(
                x, size=(NUM_FRAMES, x.shape[3], x.shape[4]),
                mode="trilinear", align_corners=False)
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = (x * self.source_std + self.source_mean - self.model_mean) / self.model_std
        return self.model(pixel_values=x).logits


def build_videomae2(num_classes: int = 35, pretrained: bool = True,
                    dropout: float = 0.3):
    return VideoMAE2Classifier(num_classes=num_classes,
                               pretrained=pretrained,
                               dropout=dropout)


if __name__ == "__main__":
    model = build_videomae2(num_classes=35, pretrained=False)
    x = torch.randn(1, 8, 3, 224, 224)
    print(model(x).shape)
