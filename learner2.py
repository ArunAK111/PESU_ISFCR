import torch
import torch.nn as nn
from torch.nn import functional as F



class Learner2(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(Learner2, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.drop_p = drop_p
        self.weight_init()
        self.vars = nn.ParameterList()
        self.filter1 = nn.LayerNorm(input_dim)

        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param)

    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars

        x1 = F.linear(x, vars[0], vars[1])
        x1 = F.relu(x1)
        x1 = F.dropout(x1, self.drop_p, training=self.training)
        x1 = F.linear(x1, vars[2], vars[3])
        x1 = F.dropout(x1, self.drop_p, training=self.training)
        x1 = F.linear(x1, vars[4], vars[5])
    
        x = self.relu(self.filter1(x))
        x2 = F.linear(x, vars[0], vars[1])
        x2 = F.relu(x2)
        x2 = F.dropout(x2, self.drop_p, training=self.training)
        x2 = F.linear(x2, vars[2], vars[3])
        x2 = F.dropout(x2, self.drop_p, training=self.training)
        x2 = F.linear(x2, vars[4], vars[5])

        x = (x1 + x2)/2.

        return torch.sigmoid(x)

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


