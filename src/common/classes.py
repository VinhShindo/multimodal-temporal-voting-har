"""Load và encode/decode class mapping."""
from pathlib import Path
import yaml


class ClassMap:
    def __init__(self, yaml_path: str | Path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.id2name = {int(k): v for k, v in cfg["classes"].items()}
        self.name2id = {v: k for k, v in self.id2name.items()}
        self.num_classes = cfg["num_classes"]

    def to_name(self, cid: int) -> str:
        return self.id2name[cid]

    def to_id(self, name: str) -> int:
        return self.name2id[name]


def load_classes(path: str | Path = "configs/classes.yaml") -> ClassMap:
    return ClassMap(path)
