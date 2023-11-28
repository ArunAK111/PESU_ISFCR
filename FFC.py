import torch
import torch.nn as nn
from torch.nn import functional as F



class Learner2(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(Learner2, self).__init__()
        self.filter1 = nn.LayerNorm(input_dim)
        self.filter2 = nn.LayerNorm(input_dim)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(drop_p)

        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 256)
        self.fc4 = nn.Linear(256, 32)
        self.fc5 = nn.Linear(32, 1)


    def forward(self, x):
        x1 = self.relu2(x)

        out = self.fc5(self.relu(self.fc4((self.relu(self.fc3(self.dropout(self.relu(self.fc2(self.relu(self.fc1(x)))))))))))
        out2 = self.fc5(self.relu(self.fc4(self.relu(self.fc3(self.dropout(self.relu(self.fc2(self.relu(self.fc1(x1))))))))))
        out = out + out2*0.2
        return torch.sigmoid(out) 



