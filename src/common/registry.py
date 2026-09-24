"""Registry đơn giản cho model/dataset."""
_REGISTRY = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"'{name}' chưa được đăng ký. Có: {list(_REGISTRY)}")
    return _REGISTRY[name]
