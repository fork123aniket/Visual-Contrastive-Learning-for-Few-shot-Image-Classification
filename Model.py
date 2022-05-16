import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self, margin):
        super(SiameseNetwork, self).__init__()
        self.margin = margin

        def reset_parameters(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.normal_(m.bias, mean=0.5, std=0.01)

        self.encoder = nn.Sequential(
                        nn.Conv2d(1, 64, 10),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 7),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(128, 128, 4),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(128, 256, 4),
                        nn.ReLU(),
                        nn.Flatten())

        self.lin1 = nn.Linear(9216, 4096)
        self.lin1.apply(reset_parameters)
        self.lin2 = nn.Linear(1, 1)
        self.lin2.apply(reset_parameters)

        self.encoder.apply(reset_parameters)

    def forward(self, input, label, k_shot):

        output = torch.zeros((k_shot + 1, input.size(1), 4096))
        scores = torch.zeros((k_shot, input.size(1)))
        losses = torch.zeros((k_shot, 1))
        for shot in range(k_shot + 1):
            output1 = self.encoder(input[shot])
            output1 = self.lin1(output1)
            output[shot] = nn.Sigmoid()(output1)

        for dist in range(k_shot):
            distance = F.pairwise_distance(output[0], output[dist + 1])
            scores[dist] = distance

        if not self.training:
            return scores

        for example in range(k_shot):
            losses[example] = torch.mean((1 - label) * torch.pow(scores[example], 2) +
                                         label * torch.pow(torch.clamp(self.margin - scores[example],
                                                                       min=0.0), 2))
        return torch.mean(losses)
