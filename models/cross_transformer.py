import torch
import torch.nn as nn
from transformers import ViTConfig
from transformers.models.vit.modeling_vit import ViTEncoder


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