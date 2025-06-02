import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.moe import MoE
from collections import OrderedDict


def activation(act):
    if act == 'RELU':
        return nn.ReLU()
    elif act == 'TANH':
        return nn.Tanh()
    elif act == 'GLU':
        return nn.GLU()
    elif act == 'ELU':
        return nn.ELU(1.3)
    elif act == 'CELU':
        return nn.CELU(1.3)
    else:
        return nn.Identity()


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            args,
            vocab_size,
            embed_dim,
            decoder_dropout,
            word_embed_dropout,
            lm_dropout,
            pe_max_len,
            att_heads,
            att_mid_dim,
            att_mid_drop,
            bifeat_act,
            bifeat_emb_act,
            bifeat_emb_drop,
            ff_dropout,
            layer_num
    ):
        super(Decoder, self).__init__()
        self.args = args
        self.att_heads = att_heads
        self.layers = nn.ModuleList([])
        self.embed_dim = embed_dim
        for i in range(layer_num):
            sublayer = DecoderLayer(
                args=args,
                embed_dim=embed_dim,
                dropout=decoder_dropout,
                att_heads=att_heads,
                att_mid_dim=att_mid_dim,
                att_mid_drop=att_mid_drop,
                bifeat_emb_act=bifeat_emb_act,
                bifeat_emb_drop=bifeat_emb_drop,
                ff_dropout=ff_dropout,
                last_layer=(i == layer_num - 1))
            self.layers.append(sublayer)

        self.dropout = nn.Dropout(word_embed_dropout)
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEncoding(embed_dim, pe_max_len)

        self.layer_norm_word = torch.nn.LayerNorm(embed_dim)
        self.generator = nn.Linear(embed_dim, vocab_size)

        self.wbil1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            activation(bifeat_act),
            torch.nn.LayerNorm(embed_dim)
        )
        self.wbil2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            activation(bifeat_act),
            torch.nn.LayerNorm(embed_dim)
        )
        self.wbi_drop = nn.Dropout(decoder_dropout)
        self.dropout_lm = nn.Dropout(lm_dropout)

        self.proj_norm = nn.Sequential(
            nn.Linear(embed_dim * (layer_num + 1), 2 * embed_dim),
            nn.GLU(),
            torch.nn.LayerNorm(embed_dim))

        self.clear_buffer()

    def init_buffer(self, batch_size):
        self.seq_len = 0
        self.x = torch.zeros((batch_size, 1, self.embed_dim)).cuda()
        for layer in self.layers:
            layer.init_buffer(batch_size)

    def clear_buffer(self):
        self.seq_len = None
        self.x = None
        for layer in self.layers:
            layer.clear_buffer()

    def apply_to_states(self, fn):
        self.x = fn(self.x)
        for layer in self.layers:
            layer.apply_to_states(fn)

    def precompute(self, encoder_out):
        p_att_feats = []
        for layer in self.layers:
            key, value2 = layer.precompute(encoder_out)
            p_att_feats.append((key, value2))
        return p_att_feats

    def forward(self, gx, prev_output_tokens, encoder_out, att_mask, seq_mask=None, p_att_feats=None, precompute=False):
        att_mask = att_mask.unsqueeze(1)
        # embed positions
        seq_len = prev_output_tokens.size(1)
        if self.seq_len is not None:
            seq_len = self.seq_len + seq_len
            self.seq_len = seq_len
            positions = self.embed_positions(seq_len)[:, -1, :].unsqueeze(1)
        else:
            positions = self.embed_positions(seq_len)

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        x = x + positions
        x = self.layer_norm_word(x)
        if self.dropout is not None:
            x = self.dropout(x)

        # decoder layers
        if self.x is None:
            x_gx = (torch.sum(x.unsqueeze(1) * seq_mask.unsqueeze(-1), -2) / torch.sum(seq_mask, -1).unsqueeze(-1))
        else:
            self.x = self.x + x
            x_gx = self.x / seq_len
        x_gx = self.wbil2(x_gx)

        gx = self.wbil1(gx)
        gx = gx.unsqueeze(1)
        gx = gx * x_gx
        gx = self.wbi_drop(gx)

        gx_arr = [gx]
        aux_loss = []
        for layerid, layer in enumerate(self.layers):
            if precompute == False:
                p_key = None
                p_value2 = None
            else:
                p_key, p_value2 = p_att_feats[layerid]
            gx, x, aux = layer(gx, x, encoder_out, att_mask, seq_mask=seq_mask, p_key=p_key, p_value2=p_value2,
                          precompute=precompute)
            gx_arr.append(gx)
            aux_loss.append(aux)

        gx = torch.cat(gx_arr, dim=-1)
        gx = self.proj_norm(gx)

        gx = self.dropout_lm(gx)
        out = self.generator(gx)

        return out


class DecoderLayer(nn.Module):
    def __init__(
            self,
            args,
            embed_dim,
            dropout,
            att_heads,
            att_mid_dim,
            att_mid_drop,
            bifeat_emb_act,
            bifeat_emb_drop,
            ff_dropout,
            last_layer=False
    ):
        super(DecoderLayer, self).__init__()
        self.args = args
        self.last_layer = last_layer
        self.word_attn = LowRank(
            embed_dim=embed_dim,
            att_heads=att_heads,
            att_mid_dim=att_mid_dim,
            att_mid_drop=att_mid_drop)
        self.word_dropout = nn.Dropout(dropout)

        self.cross_att = LowRank(
            embed_dim=embed_dim,
            att_heads=att_heads,
            att_mid_dim=att_mid_dim,
            att_mid_drop=att_mid_drop)
        self.cross_dropout = nn.Dropout(dropout)
        self.layer_norm_cross = torch.nn.LayerNorm(embed_dim)

        if self.last_layer == False:
            self.bifeat_emb = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                activation(bifeat_emb_act),
                nn.Dropout(bifeat_emb_drop)
            )
            self.layer_norm_x = torch.nn.LayerNorm(embed_dim)

            if self.args.ff_type == 'linear':
                self.ff_layer = FeedForwardBlock(
                    embed_dim=embed_dim,
                    ffn_embed_dim=embed_dim * 4,
                    relu_dropout=ff_dropout,
                    dropout=ff_dropout)
            else:
                self.ff_layer = MoE(
                dim = embed_dim,
                num_experts = 16,               # increase the experts (# parameters) of your model without increasing computation
                hidden_dim = embed_dim * 4,           # size of hidden dimension in each expert, defaults to 4 * dimension
                activation = nn.LeakyReLU,      # use your preferred activation, will default to GELU
                second_policy_train = 'random', # in top_2 gating, policy for whether to use a second-place expert
                second_policy_eval = 'random',  # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
                second_threshold_train = 0.2,
                second_threshold_eval = 0.2,
                capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
                capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
                loss_coef = 0.5                # multiplier on the auxiliary expert balancing auxiliary loss
            )

        self.layer_norm_gx = torch.nn.LayerNorm(embed_dim)

    def apply_to_states(self, fn):
        self.word_attn.apply_to_states(fn)

    def init_buffer(self, batch_size):
        self.word_attn.init_buffer(batch_size)

    def clear_buffer(self):
        self.word_attn.clear_buffer()

    def precompute(self, encoder_out):
        key, value2 = self.cross_att.precompute(encoder_out, encoder_out)
        return key, value2

    def forward(
            self,
            gx,
            x,
            encoder_out,
            att_mask,
            seq_mask,
            p_key=None,
            p_value2=None,
            precompute=False
    ):
        word_x = x
        residual = x
        x = self.word_attn.forward2(
            query=gx,
            key=x,
            mask=seq_mask,
            value1=gx,
            value2=x)
        x = self.word_dropout(x)
        x = residual + x

        residual = x
        x = self.layer_norm_cross(x)
        x = self.cross_att.forward2(
            query=x,
            key=encoder_out if precompute == False else p_key,
            mask=att_mask,
            value1=x,
            value2=encoder_out if precompute == False else p_value2,
            precompute=precompute)
        x = self.cross_dropout(x)
        gx = residual + x
        gx = self.layer_norm_gx(gx)

        if self.last_layer == False:
            x_ = torch.cat([gx, word_x], dim=-1)
            x = self.bifeat_emb(x_) + word_x
            x = self.layer_norm_x(x)

            if self.args.ff_type == 'linear':
                x = self.ff_layer(x)
                aux_loss = 0
            else:
                x, aux_loss = self.ff_layer(x)

        else:
            x = None
            aux_loss = 0
        return gx, x, aux_loss


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, relu_dropout, dropout):
        super(FeedForwardBlock, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.layer_norms = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.layer_norms(x)
        return x


class LowRank(nn.Module):
    def __init__(self, embed_dim, att_heads, att_mid_dim, att_mid_drop, act_type="CELU"):
        super(LowRank, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        output_dim = 2 * embed_dim if act_type == 'GLU' else embed_dim

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = activation(act_type)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_q = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = activation(act_type)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_k = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = activation(act_type)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v1 = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = activation(act_type)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v2 = nn.Sequential(*sequential)

        self.attn_net = SCAtt(att_mid_dim, att_mid_drop)
        self.clear_buffer()

    def apply_to_states(self, fn):
        self.buffer_keys = fn(self.buffer_keys)
        self.buffer_value2 = fn(self.buffer_value2)

    def init_buffer(self, batch_size):
        self.buffer_keys = torch.zeros((batch_size, self.num_heads, 0, self.head_dim)).cuda()
        self.buffer_value2 = torch.zeros((batch_size, self.num_heads, 0, self.head_dim)).cuda()

    def clear_buffer(self):
        self.buffer_keys = None
        self.buffer_value2 = None

    # query -- batch_size * qdim
    # value -- batch_size * att_num * vdim
    def forward(self, query, key, mask, value1, value2, precompute=False):
        batch_size = query.size()[0]
        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)

        q = q.reshape(batch_size, self.num_heads, self.head_dim)
        v1 = v1.reshape(batch_size, self.num_heads, self.head_dim)

        if precompute == False:
            key = key.reshape(-1, key.size()[-1])
            value2 = value2.reshape(-1, value2.size()[-1])
            k = self.in_proj_k(key)
            v2 = self.in_proj_v2(value2)
            k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v2 = v2.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k = key
            v2 = value2

        attn_map = q.unsqueeze(-2) * k
        attn = self.attn_net(attn_map, mask, v1, v2)
        attn = attn.reshape(batch_size, self.num_heads * self.head_dim)
        return attn

    # query -- batch_size * seq_num * qdim
    # value -- batch_size * att_num * vdim
    def forward2(self, query, key, mask, value1, value2, precompute=False):
        batch_size = query.size()[0]
        query = query.reshape(-1, query.size()[-1])
        value1 = value1.reshape(-1, value1.size()[-1])

        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)

        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = v1.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if precompute == False:
            key = key.reshape(-1, key.size()[-1])
            value2 = value2.reshape(-1, value2.size()[-1])
            k = self.in_proj_k(key)
            v2 = self.in_proj_v2(value2)
            k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v2 = v2.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            if self.buffer_keys is not None and self.buffer_value2 is not None:
                self.buffer_keys = torch.cat([self.buffer_keys, k], dim=2)
                self.buffer_value2 = torch.cat([self.buffer_value2, v2], dim=2)
                k = self.buffer_keys
                v2 = self.buffer_value2
        else:
            k = key
            v2 = value2

        attn_map = q.unsqueeze(-2) * k.unsqueeze(-3)
        attn = self.attn_net.forward(attn_map, mask, v1, v2).transpose(1, 2).contiguous()
        attn = attn.reshape(batch_size, -1, self.num_heads * self.head_dim)
        return attn

    def precompute(self, key, value2):
        batch_size = value2.size()[0]
        key = key.reshape(-1, key.size()[-1])
        value2 = value2.reshape(-1, value2.size()[-1])

        k = self.in_proj_k(key)
        v2 = self.in_proj_v2(value2)

        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = v2.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        return k, v2


class BasicAtt(nn.Module):
    def __init__(self, mid_dims, mid_dropout):
        super(BasicAtt, self).__init__()

        sequential = []
        for i in range(1, len(mid_dims) - 1):
            sequential.append(nn.Linear(mid_dims[i - 1], mid_dims[i]))
            sequential.append(nn.ReLU())
            if mid_dropout > 0:
                sequential.append(nn.Dropout(mid_dropout))
        self.attention_basic = nn.Sequential(*sequential) if len(sequential) > 0 else None
        self.attention_last = nn.Linear(mid_dims[-2], mid_dims[-1])

    def forward(self, att_map, att_mask, value1, value2):
        if self.attention_basic is not None:
            att_map = self.attention_basic(att_map)
        attn_weights = self.attention_last(att_map)
        attn_weights = attn_weights.squeeze(-1)
        if att_mask is not None:
            attn_weights = attn_weights.masked_fill(att_mask.unsqueeze(1) == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn = torch.matmul(attn_weights.unsqueeze(-2), value2).squeeze(-2)
        return attn


class SCAtt(BasicAtt):
    def __init__(self, mid_dims, mid_dropout):
        super(SCAtt, self).__init__(mid_dims, mid_dropout)
        self.attention_last = nn.Linear(mid_dims[-2], 1)
        self.attention_last2 = nn.Linear(mid_dims[-2], mid_dims[-1])

    def forward(self, att_map, att_mask, value1, value2):
        if self.attention_basic is not None:
            att_map = self.attention_basic(att_map)

        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)
            att_mask_ext = att_mask.unsqueeze(-1)
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, -2)
        else:
            att_map_pool = att_map.mean(-2)

        alpha_spatial = self.attention_last(att_map)
        alpha_channel = self.attention_last2(att_map_pool)
        alpha_channel = torch.sigmoid(alpha_channel)

        alpha_spatial = alpha_spatial.squeeze(-1)
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask == 0, np.half(-1e9))
        alpha_spatial = F.softmax(alpha_spatial, dim=-1)

        if len(alpha_spatial.shape) == 4:  # batch_size * head_num * seq_num * seq_num (for xtransformer)
            value2 = torch.matmul(alpha_spatial, value2)
        else:
            value2 = torch.matmul(alpha_spatial.unsqueeze(-2), value2).squeeze(-2)

        attn = value1 * value2 * alpha_channel
        return attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(max_len * 2.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x_size):
        return self.pe[:, :x_size]


class RotaryEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, device):
        t = torch.arange(max_seq_len, device = device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


if __name__ == "__main__":
    from vision.x_linear_encoder import Encoder
    encoder = Encoder(
            embed_dim = 768,
            dropout = 0.5,
            att_heads=8,
            att_mid_dim=[96, 64, 96],
            att_mid_drop=0.1,
            bifeat_emb_act="RELU",
            bifeat_emb_drop=0.3,
            ff_dropout=0.1,
            layer_num=4)

    encoder = encoder.to('cuda')

    x = torch.rand(2, 49, 768).cuda()
    mask = torch.ones(2, 49).cuda()
    seq = torch.ones(2, 30).long().cuda()
    seq_mask = torch.ones(2, 30, 30).cuda()
    out = encoder(x, mask)
    encoder_out, gx = out['last_hidden_state'], out['pooler_output']
    decoder = Decoder(
            vocab_size=1000,
            embed_dim=768,
            decoder_dropout=0.1,
            word_embed_dropout=0.1,
            lm_dropout=0.1,
            pe_max_len=5000,
            att_heads=8,
            att_mid_dim=[96, 64, 96],
            att_mid_drop=0.5,
            bifeat_act='RELU',
            bifeat_emb_act='CELU',
            bifeat_emb_drop=0.3,
            ff_dropout=0.5,
            layer_num=6)
    decoder = decoder.to('cuda')
    decoder_out = decoder(gx, seq, encoder_out, mask, seq_mask)

