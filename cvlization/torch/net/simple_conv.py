from torch import nn


class SimpleConv(nn.Module):
    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.relu = nn.ReLU()
        convs = []
        for out_channels in [32, 64]:
            convs.append(
                nn.LazyConv2d(
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                )
            )
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        y = x
        for c in self.convs:
            y = c(y)
            y = self.relu(y)
        # flatten = nn.Flatten()
        # y = flatten(y)
        return y
