import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class DropoutModel(BaseModel):
    def __init__(self, p_dropout):
        """
        p_dropout: probability of dropout, float in range [0.0, 1.0]

        'We use NNs with either 4 or 5 hidden layers and 1024 hidden units.
        We use either ReLU non-linearities or TanH non-linearities in each
        network, and use dropout probabilities of either 0.1 or 0.2. Exact
        experiment setup is given in section E.1 in the appendix.'
        https://arxiv.org/pdf/1506.02142.pdf, Section 5.1
        """
        super().__init__()
        self.fc1 = nn.Linear(1, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(p=p_dropout)

        for m in [self.fc1, self.fc2, self.fc3, self.fc4]:
            self.initialize_weights(m)

    def forward(self, x):
        """
        'With dropout, we sample binary variables for every input point and
        for every network unit in each layer (apart from the last one).'
        https://arxiv.org/pdf/1506.02142.pdf, Section 3
        """
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)         
        x = F.relu(self.fc4(x))

    def initialize_weights(m):
        """
        'We initialize the bias at 0 and initialize the weights uniformly
        from [-sqrt(3/fan-in), +sqrt(3/fan-in)].'
        https://arxiv.org/pdf/1506.02157.pdf, Section E.1
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu') # Does this do sqrt(3/fan-in)?
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    