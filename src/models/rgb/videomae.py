"""VideoMAE ViT-base video classifier backed by Hugging Face Transformers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CHECKPOINT = "MCG-NJU/videomae-base-finetuned-kinetics"
NUM_FRAMES = 16


def _load_checkpoint(model: nn.Module) -> None:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    checkpoint_path = hf_hub_download(
        repo_id=DEFAULT_CHECKPOINT,
        filename="model.safetensors",
    )
    source = load_file(checkpoint_path, device="cpu")
    target = model.state_dict()
    for index in range(12):
        prefix = f"videomae.encoder.layer.{index}.attention.attention."
        query_bias = source.pop(prefix + "q_bias")
        value_bias = source.pop(prefix + "v_bias")
        source[prefix + "query.bias"] = query_bias
        key_bias = prefix + "key.bias"
        if key_bias in target:
            source[key_bias] = torch.zeros_like(query_bias)
        source[prefix + "value.bias"] = value_bias
    model.load_state_dict(source, strict=True)


class VideoMAEClassifier(nn.Module):
    def __init__(self, num_classes: int = 35, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        try:
            from transformers import VideoMAEConfig, VideoMAEForVideoClassification
        except ImportError as error:
            raise ImportError(
                "VideoMAE requires transformers; install project requirements."
            ) from error

        if pretrained:
            config = VideoMAEConfig.from_pretrained(DEFAULT_CHECKPOINT)
            self.model = VideoMAEForVideoClassification(config)
            _load_checkpoint(self.model)
        else:
            config = VideoMAEConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                num_frames=NUM_FRAMES,
                tubelet_size=2,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                num_labels=num_classes,
                use_mean_pooling=True,
            )
            self.model = VideoMAEForVideoClassification(config)

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


def build_videomae(num_classes: int = 35, pretrained: bool = True,
                   dropout: float = 0.3):
    return VideoMAEClassifier(num_classes=num_classes,
                              pretrained=pretrained,
                              dropout=dropout)


if __name__ == "__main__":
    model = build_videomae(num_classes=35, pretrained=False)
    x = torch.randn(1, 8, 3, 224, 224)
    print(model(x).shape)
