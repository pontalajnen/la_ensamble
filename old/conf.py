@dataclass
class Config:
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0001
    num_workers: int = 2
    dataset: str = "CIFAR10"
    model: str = "ResNet18"
    device: str = "cuda"
    seed: int