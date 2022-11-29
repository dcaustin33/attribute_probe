import torch.nn as nn

class attribute_mlp(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.classifier1 = nn.Sequential(nn.Linear(input_dim, 1024),
                                         nn.ReLU(), 
                                         nn.BatchNorm1d(1024),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(1024),
                                        nn.Linear(1024, output_dim))
        self.classifier2 = nn.Sequential(nn.Linear(input_dim, output_dim))
        



    def forward(self, attributes):
        output = {}
        output['classifier1'] = self.classifier1(attributes)
        output['classifier2'] = self.classifier2(attributes)
        return output