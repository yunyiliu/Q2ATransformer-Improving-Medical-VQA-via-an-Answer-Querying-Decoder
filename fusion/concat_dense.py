import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size, dropout=0.2):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(input_size, intermediate_size)
        self.bn1 = nn.BatchNorm1d(intermediate_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_size, output_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    # @torchsnooper.snoop()
    def forward(self, inputs):
        x = self.fc1(inputs) #x : 4, 256 inputs: [4, 1792]
        x = self.bn1(x) #[4, 256]
        x = self.act1(x) #[4, 256]
        x = self.fc2(x) #[4, 256]
        return x


class LinearFusionLayer(nn.Module):

    def __init__(self, lang_input_size, visn_input_size, intermediate_size, output_size, dropout=0.2):
        super(LinearFusionLayer, self).__init__()
        self.projector = Projector(lang_input_size + visn_input_size, intermediate_size, output_size, dropout)

    def forward(self, lang_feats, visn_feats):
        inputs = torch.cat([lang_feats, visn_feats], dim=-1)# inputs [4, 1792] lang [4, 768] visn [4, 1024]
        outputs = self.projector(inputs) # 1792 - 256 - 768
        return outputs #[4, 768]


class AttLinearFusionLayer(nn.Module):
    def __init__(self, visn_input_size, intermediate_size, output_size, dropout=0.2):
        super(AttLinearFusionLayer, self).__init__()
        self.projector = Projector(visn_input_size, intermediate_size, output_size, dropout)

    # @torchsnooper.snoop()
    def forward(self, lang_feats, visn_feats):
        inputs = torch.cat([visn_feats, lang_feats], dim=1)
        outputs = self.projector(inputs)
        return outputs
