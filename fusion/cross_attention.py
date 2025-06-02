import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.lxmert.modeling_lxmert import LxmertXLayer
from transformers import ViTConfig
from transformers.models.vit.modeling_vit import ViTEncoder


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

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.fc2(x)

        return x


class LxmertXFusionLayer(nn.Module):

    def __init__(
            self, num_attention_heads, num_hidden_layers, intermediate_size, hidden_size,
            input_lang_size, input_visn_size, fc_intermediate_size, output_size, dropout=0.2
    ):
        """
        Args:
            num_attention_heads:
            num_hidden_layers:
            intermediate_size:
            hidden_size:
            input_lang_size:
            input_visn_size:
            fc_intermediate_size:
            output_size:
            dropout:
        """
        super(LxmertXFusionLayer, self).__init__()

        lxrt_config = BertConfig(
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size
        )
        self.x_layers = nn.ModuleList(
            [LxmertXLayer(lxrt_config) for _ in range(lxrt_config.num_hidden_layers)]
        )

        self.input_lang_fc = nn.Linear(input_lang_size, hidden_size)
        self.input_visn_fc = nn.Linear(input_visn_size, hidden_size)
        self.projector = Projector(2 * hidden_size, fc_intermediate_size, output_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.input_lang_fc.weight)
        nn.init.xavier_uniform_(self.input_visn_fc.weight)

    def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask):
        """
        Args:
            lang_feats:
            lang_attention_mask:
            visn_feats:
            visn_attention_mask:
        Returns:
        """
        lang_attention_mask = lang_attention_mask.float().unsqueeze(1).unsqueeze(2)
        visn_attention_mask = visn_attention_mask.float().unsqueeze(1).unsqueeze(2)
        extended_lang_mask = (1 - lang_attention_mask) * -10000
        extended_visn_mask = (1 - visn_attention_mask) * -10000

        lang_feats = self.dropout(self.act(self.input_lang_fc(lang_feats)))
        visn_feats = self.dropout(self.act(self.input_visn_fc(visn_feats)))

        for layer_module in self.x_layers:
            lang_feats, visn_feats = layer_module(
                lang_feats, extended_lang_mask, visn_feats, extended_visn_mask
            )

        cat = torch.cat([lang_feats[:, 0, :], visn_feats[:, 0, :]], dim=1)
        feat = self.projector(cat)

        return feat, lang_feats, visn_feats


class CrossFusionLayer(nn.Module):
    def __init__(self, num_hidden_layers):
        super(CrossFusionLayer, self).__init__()
        config = ViTConfig()
        config.num_hidden_layers = num_hidden_layers
        self.encoder = ViTEncoder(config)

    def forward(self, vision_feat, lang_feat):
        feat = torch.cat([vision_feat, lang_feat], 1)
        out = self.encoder(feat, return_dict=False)[0]
        return out


if __name__ == "__main__":
    encoder = CrossFusionLayer(num_hidden_layers=3).cuda()

    visn_feat = torch.rand(4, 20, 768).cuda()
    lang_feat = torch.rand(4, 49, 768).cuda()

    out = encoder(visn_feat, lang_feat)
    print(out.shape)
