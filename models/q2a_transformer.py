# swin transformer end-to-end without pretrain
import json


import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertConfig,
    BertTokenizer
)
from fusion import LinearFusionLayer
from lightning_tools.optimizers import config_nodecay_optimizer, config_optimizer, config_decoder_optimizer
from layers.losses import MatchingLayer
from utils.metrics import Recorder
from vision import swin_transformer
from models.transformer import build_transformer
from models.cross_transformer import CrossFusionLayer
import models
import models.aslloss
import math

class MultiModal(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        # ============================ Save configs to yaml file
        self.save_hyperparameters(args)
        self.save_hyperparameters(kwargs)
        self.save_hyperparameters()
        # ============================ Language ============================
        bert_config = BertConfig.from_pretrained(self.hparams.bert_dir)
        bert_config.hidden_dropout_prob = self.hparams.bert_hidden_dropout_prob
        self.bert_model = BertModel.from_pretrained(
            self.hparams.bert_dir, config=bert_config, cache_dir=self.hparams.bert_cache_dir
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            self.hparams.bert_dir, use_fast=True, cache_dir=self.hparams.bert_cache_dir
        )
        # ============================ vision ============================
        self.vision_model = swin_transformer.from_pretrained(
            self.hparams.swin_transformer_yaml, self.hparams.swin_transformer_ckpt
        )
        # ============================ cross encoder ============================
        self.fc_fusion_layer = LinearFusionLayer(
            self.hparams.lang_feat_size, self.hparams.vision_feat_size, self.hparams.fc_size,
            self.hparams.metric_ebd_size,
            dropout=self.hparams.dropout
        ) #[256 ,768]
        
        self.cross_fusion = CrossFusionLayer(self.hparams.metric_ebd_size) 
        self.matching_layer = MatchingLayer(self.hparams.tau)
        
        self.cls_loss = nn.BCEWithLogitsLoss()
        #============================ answer embedding ============================
        # args -> self.hparam
        self.transformer = build_transformer(self.hparam) #2048
        self.num_class = self.hparams.num_class #557
        hidden_dim = self.transformer.d_model  #2048

        # self.input_proj = nn.Conv2d(self.fc_fusion_layer.size(0), hidden_dim, kernel_size=1)
        
        self.input_proj = nn.Conv2d(self.hparams.metric_ebd_size, 
                                    hidden_dim, 
                                    kernel_size=1)

        self.query_embed = nn.Embedding(self.num_class, hidden_dim) #[557, 2048]
        self.projector = nn.Linear(1024, 768)
        self.projector2 = nn.Linear(768, 2048)
        self.fc = GroupWiseLinear(self.num_class, hidden_dim, bias=True) #bias【1，557】w:1, 557, 2048
        # ============================ Metrics ops ===========================
        self.train_metric = Recorder()
        self.val_metric = Recorder()
        # ===============================loss================================
        self.criterion = models.aslloss.AsymmetricLossOptimized(
            gamma_neg=self.hparams.gamma_neg, gamma_pos=self.hparams.gamma_pos,
            clip=self.hparams.loss_clip,
            disable_torch_grad_focal_loss=self.hparams.dtgfl,
            eps=self.hparams.eps,
        )
        # self.cls_loss = nn.CrossEntropyLoss()


    def forward(self, batch):
        # ============================ language encoder ============================
        question_bert_outputs = self.bert_model(batch['question_input_ids'], batch['question_mask'], batch['question_token_type_ids'])
        question_last_hidden_state, question_bert_pooler = question_bert_outputs["last_hidden_state"], question_bert_outputs["pooler_output"]
        #last_h_s:[4, 20, 768]  pooler:[4, 768]
        # ============================ vision encoder ============================
        visn_outputs = self.vision_model(batch['pixel_values']) #last_h_s:[4, 49, 1024] pooler:[4, 1024]pos:[4, 3136, 128]
        visn_last_hidden_state, visn_pooler, pos = visn_outputs["last_hidden_state"], visn_outputs["pooler_output"], visn_outputs["patch_embed"]
        # visn_pooler: [4, 1024], _: [4, 49, 1024]

        visn_last_hidden_state = self.projector(visn_last_hidden_state) #[4, 49, 768]
        # ============================ fusion ============================
        fusion_res = self.cross_fusion(question_last_hidden_state, visn_last_hidden_state) #[4, 69, 768]
        query_input = self.query_embed.weight #[557, 2048]
        fusion_res = self.projector2(fusion_res) #[4, 69, 2048]
        hs = self.transformer(fusion_res, query_input, None)[0] #[1, 4, 557, 2048]
        predictions = self.fc(hs[-1]) #hidden_dim: 2048 num_class: 557 output： [4, 557]

        cls_loss = self.criterion(predictions, batch['target_ids'].long())

        to_return = {'loss': cls_loss, 'predictions': predictions, 'labels': batch['target_ids']}
        return to_return

    def training_step(self, batch, batch_idx, optimizer_idx):
        result = self(batch)
        return result
    
    def training_step_end(self, training_step_outputs):
        """
        PL Hook，把每个worker的train_step收集起来
        :param training_step_outputs: 一个worker(dict)，否则list(dict)
        :return:
        """
        if isinstance(training_step_outputs, list):  # 多worker
            loss2list = dict()
            for batch in training_step_outputs:
                for key in batch:
                    if "loss" in key:
                        loss2list.setdefault(key, list()).append(batch[key])

            loss2mean = dict()
            for key in loss2list:
                loss2mean[key] = torch.stack(loss2list[key]).mean()
        else:
            loss2mean = dict()
            for key in training_step_outputs:
                if "loss" in key:
                    loss2mean[key] = training_step_outputs[key]
        to_log = {}
        to_log.update(loss2mean)
        self.log_dict(to_log, prog_bar=True)
        self.train_metric.reset()
        self.print(loss2mean["loss"])
        return loss2mean["loss"]
        
    def validation_step(self, batch, batch_idx):
        results = self(batch)
        return results

    def validation_epoch_end(self, outputs):
        """
        PL HOOK,每个val epoch end，收集recorder的结果，并且汇总，存储checkpoints
        Args:
            outputs:
        Returns:
        """
        preds = []
        labels = []
        # print(outputs)
        for i in range(len(outputs)):
            p = outputs[i]['predictions']
            preds.append(p)
            l = outputs[i]['labels']
            labels.append(l)
        preds = torch.cat(preds).squeeze() #[338, 557]
        labels = torch.cat(labels) #[338]
        _, preds = torch.max(preds, dim=1)
        preds = (nn.Sigmoid()(preds) > 0.5).byte() #[338, 557]
        # print(preds.shape)
        
        acc = (preds == labels).sum()/len(labels)
        self.print(f"ACC:{acc}")
        return outputs


    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        normal_optimizer, normal_scheduler = config_optimizer(
            list(self.fc_fusion_layer.parameters()) + list(self.projector.parameters()),
            self.hparams.learning_rate, self.hparams.warmup_steps, self.hparams.max_steps,
            name="lr"
        )

        vision_optimizer, vision_scheduler = config_nodecay_optimizer(
            self.vision_model, self.hparams.vision_learning_rate, self.hparams.vision_warmup_steps,
            self.hparams.max_steps,
            name="vision_lr"
        )

        bert_optimizer, bert_scheduler = config_nodecay_optimizer(
            self.bert_model, self.hparams.bert_learning_rate, self.hparams.bert_warmup_steps, self.hparams.max_steps,
            name="bert_lr"
        )

        decoder_optimizer, decoder_scheduler = config_decoder_optimizer(
            list(self.transformer.parameters()), self.hparams.decoder_lr, self.hparams.decoder_warmup_steps,
            self.hparams.model_size,
            name='decoder_lr'
        )

        opts = [bert_optimizer, normal_optimizer, vision_optimizer, decoder_optimizer]
        schedulers = [bert_scheduler, normal_scheduler, vision_scheduler, decoder_scheduler]
        return opts, schedulers

    @staticmethod
    def bert_pool(hidden_state, masks):
        """
        cat average pool and [cls] token embeddings
        Args:
            hidden_state:
            masks:
        Returns:
        """
        dtype = hidden_state.dtype
        bert_mean_pooler = (masks.unsqueeze(-1).to(dtype) * hidden_state).sum(1) / (
                    masks.sum(dim=1, keepdim=True) + 1e-5).to(dtype)

        cat = torch.cat([bert_mean_pooler, hidden_state[:, 0]], dim=1)
        return cat

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad()

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            # 从均匀分布中抽样数值进行填充
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x