import torch, torch.nn as nn


class Inception_Block(nn.Module):
    """
    Adapted from the TimesNet paper
    https://arxiv.org/pdf/2210.02186
    """

    def __init__(self, in_channels, out_channels, num_kernels):
        super(Inception_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = [
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            for i in range(self.num_kernels)
        ]
        self.kernels = nn.ModuleList(kernels)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = [self.kernels[i](x) for i in range(self.num_kernels)]
        res = torch.stack(res_list, dim=-1)
        return res.mean(-1)


class EyetrackBranch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_filters: int,
    ) -> None:
        super(EyetrackBranch, self).__init__()
        self.conv = nn.Sequential(
            Inception_Block(in_channels, base_filters, num_kernels=4),
            nn.GELU(),
            Inception_Block(base_filters, in_channels, num_kernels=4),
        )
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply Inception Block with (x, y) as channels
        x = self.conv(x)
        # average across trials
        x = x.mean(dim=2)
        # concat (x, y)
        return self.flatten(x)


class EyeTrackModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_filters: int,
        seq_len: int,
        n_classes: int,
    ) -> None:
        super(EyeTrackModel, self).__init__()

        self.pos_branch = EyetrackBranch(in_channels, base_filters)
        self.neg_branch = EyetrackBranch(in_channels, base_filters)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(seq_len * 4, 128),
            nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x_pos, x_neg = x[..., 0], x[..., 1]

        pos = self.pos_branch(x_pos)
        neg = self.neg_branch(x_neg)

        x = torch.cat([pos, neg], dim=-1)
        return self.classifier(x)


if __name__ == "__main__":
    sample = torch.randn(64, 2, 30, 300, 2)
    model = EyeTrackModel(in_channels=2, base_filters=32, seq_len=300, n_classes=2)
    print(model(sample.float()).shape)
