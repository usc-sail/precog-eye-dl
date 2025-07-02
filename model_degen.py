import torch, torch.nn as nn


class DeGenModel(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(DeGenModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x.float())


if __name__ == "__main__":
    sample = torch.randn(64, 2)
    model = DeGenModel(n_classes=2)
    print(model(sample.float()).shape)
