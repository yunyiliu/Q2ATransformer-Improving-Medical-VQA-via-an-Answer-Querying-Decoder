# swin transformer end-to-end without pretrain
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertConfig,
    BertTokenizer
)
from fusion import LinearFusionLayer
from utils.scorer import Scorer
import torch.nn.functional as F
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from torch.autograd import Variable
from lightning_tools.optimizers import config_nodecay_optimizer, config_decoder_optimizer, config_optimizer
from layers.losses import CrossEntropy, RewardCriterion
from utils.metrics import Recorder
from vision import swin_transformer
from models.x_tranformer_decoder import Decoder
import numpy as np
from utils import utils
import os
import json


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
        )

        # self.att_fusion_layer = AttLinearFusionLayer(
        #     self.hparams.lang_feat_size, self.hparams.fc_size,
        #     self.hparams.metric_ebd_size,
        #     dropout=self.hparams.dropout
        # )

        # ============================ decoder ============================
        self.decoder = Decoder(
                args,
                vocab_size=self.hparams.bert_vocab_size,
                embed_dim=self.hparams.embed_dim,
                decoder_dropout=self.hparams.decoder_dropout,
                word_embed_dropout=self.hparams.word_embed_dropout,
                lm_dropout=self.hparams.lm_dropout,
                pe_max_len=self.hparams.pe_max_len,
                att_heads=self.hparams.att_heads,
                att_mid_dim=self.hparams.att_mid_dim,
                att_mid_drop=self.hparams.att_mid_drop,
                bifeat_act=self.hparams.bifeat_act,
                bifeat_emb_act=self.hparams.bifeat_emb_act,
                bifeat_emb_drop=self.hparams.bifeat_emb_drop,
                ff_dropout=self.hparams.ff_dropout,
                layer_num=self.hparams.layer_num)
        # ============================ RL =====================
        self.scorer = Scorer(self.hparams)
        # ============================ Loss Ops
        self.cross_entropy = CrossEntropy()
        # ============================ Metrics ops
        self.train_metric = Recorder()
        self.val_metric = Recorder()


    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    def _shift_right_mask(self, input_ids):
        "Mask out subsequent positions."
        seq_mask = (input_ids > 0).type(torch.cuda.IntTensor)
        # for i, j in enumerate(torch.sum(seq_mask, 1)-1):
        #     seq_mask[i][j] = 0
        seq_mask[:, 0] += 1
        seq_mask = seq_mask.unsqueeze(-2)
        size = input_ids.size(-1)
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask) == 0
        seq_mask = seq_mask & subsequent_mask.to(seq_mask)
        seq_mask = seq_mask.type(torch.cuda.FloatTensor)
        return seq_mask

    def forward(self, batch):
        # language encoder
        question_bert_outputs = self.bert_model(batch['question_input_ids'], batch['question_mask'], batch['question_token_type_ids'])
        question_bert_hidden_state, question_bert_pooler = question_bert_outputs["last_hidden_state"], question_bert_outputs["pooler_output"]

        # vision encoder
        visn_outputs = self.vision_model(batch['pixel_values'])
        visn_hidden_state, visn_pooler = visn_outputs["last_hidden_state"], visn_outputs["pooler_output"]

        # fusion
        # fusion_hidden_state = self.att_fusion_layer(visn_hidden_state, question_bert_hidden_state)
        fusion_hidden_state = visn_hidden_state
        fusion_pooler = self.fc_fusion_layer(visn_pooler, question_bert_pooler)

        seq_mask = self._shift_right_mask(batch['answer_input_ids'])
        # fusion_mask = torch.cat([torch.ones(size=visn_hidden_state.size()[:-1], device=batch['answer_input_ids'].device,
        #                                 dtype=torch.float32), batch['question_mask']], 1)
        att_mask = torch.ones(size=visn_hidden_state.size()[:-1], device=batch['answer_input_ids'].device, dtype=torch.float32)
        logit = self.decoder(fusion_pooler, batch['answer_input_ids'], fusion_hidden_state, att_mask, seq_mask)

        ce_loss = self.cross_entropy(logit, batch['target_ids'])

        to_return = {'loss': ce_loss}

        return to_return

    def get_logprobs_state(self, wt, state, encoder_out, att_mask, gx, p_att_feats):
        att_mask = att_mask[:, :encoder_out.shape[1]]
        if state is None:
            ys = wt.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], wt.unsqueeze(1)], dim=1)
        subsequent_mask = np.triu(np.ones((1, ys.size(1), ys.size(1))), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask) == 0
        seq_mask = subsequent_mask.to(encoder_out.device).type(torch.cuda.FloatTensor)[:, -1, :].unsqueeze(1)
        decoder_out = self.decoder(gx, ys[:, -1].unsqueeze(-1), encoder_out, att_mask, seq_mask, p_att_feats, True).squeeze(1)
        logprobs = F.log_softmax(decoder_out, dim=-1)
        return logprobs, [ys.unsqueeze(0)]

    def decode(self, batch, greedy_decode=True):
        # language encoder
        question_bert_outputs = self.bert_model(batch['question_input_ids'], batch['question_mask'], batch['question_token_type_ids'])

        # vision encoder
        visn_outputs = self.vision_model(batch['pixel_values'])
        visn_hidden_state, visn_pooler = visn_outputs["last_hidden_state"], visn_outputs["pooler_output"]

        # fusion
        # visn_hidden_state = self.att_fusion_layer(visn_hidden_state, question_bert_hidden_state)
        visn_pooler = self.fc_fusion_layer(visn_pooler, question_bert_pooler)

        batch_size = visn_hidden_state.size(0)
        p_att_feats = self.decoder.precompute(visn_hidden_state)
        self.decoder.init_buffer(batch_size)

        state = None
        sents = Variable(torch.zeros((batch_size, self.hparams.bert_answer_max_length), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, self.hparams.bert_answer_max_length).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
        # att_mask = torch.cat([torch.ones(size=visn_hidden_state.size()[:-1], device=batch['answer_input_ids'].device,
        #                                 dtype=torch.float32), batch['question_mask']], 1)
        att_mask = torch.ones(size=visn_hidden_state.size()[:-1], device=batch['answer_input_ids'].device, dtype=torch.float32)
        for t in range(self.hparams.bert_answer_max_length):
            logprobs_t, state = self.get_logprobs_state(wt, state, visn_hidden_state, att_mask, visn_pooler, p_att_feats)
            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:, t] = wt
            logprobs[:, t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        self.decoder.clear_buffer()
        return sents, logprobs


    def select(self, batch_size, beam_size, t, candidate_logprob):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob


    def _expand_state(self, batch_size, beam_size, cur_beam_size, selected_beam):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([batch_size, beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s
        return fn


    def decode_beam(self, batch):
        beam_size = self.hparams.beam_size
        # language encoder
        question_bert_outputs = self.bert_model(batch['question_input_ids'], batch['question_mask'], batch['question_token_type_ids'])
        question_bert_hidden_state, question_bert_pooler = question_bert_outputs["last_hidden_state"], question_bert_outputs["pooler_output"]

        # vision encoder
        visn_outputs = self.vision_model(batch['pixel_values'])
        visn_hidden_state, visn_pooler = visn_outputs["last_hidden_state"], visn_outputs["pooler_output"]

        # fusion
        # visn_hidden_state = self.att_fusion_layer(visn_hidden_state, question_bert_hidden_state)
        visn_pooler = self.fc_fusion_layer(visn_pooler, question_bert_pooler)

        # att_mask = torch.cat([torch.ones(size=visn_hidden_state.size()[:-1], device=batch['answer_input_ids'].device,
        #                                 dtype=torch.float32), batch['question_mask']], 1)
        att_mask = torch.ones(size=visn_hidden_state.size()[:-1], device=batch['answer_input_ids'].device, dtype=torch.float32)
        batch_size = visn_hidden_state.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        p_att_feats = self.decoder.precompute(visn_hidden_state)

        state = None
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())

        outputs = []
        self.decoder.init_buffer(batch_size)
        for t in range(self.hparams.bert_answer_max_length):
            cur_beam_size = 1 if t == 0 else beam_size
            word_logprob, state = self.get_logprobs_state(wt, state, visn_hidden_state, att_mask, visn_pooler, p_att_feats)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)
            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            # selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]
            selected_words = selected_idx % candidate_logprob.shape[-1]
            self.decoder.apply_to_states(self._expand_state(batch_size, beam_size, cur_beam_size, selected_beam))
            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                                             selected_beam.unsqueeze(-1).expand(batch_size, beam_size,
                                                                                word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                visn_hidden_state = utils.expand_tensor(visn_hidden_state, beam_size)
                visn_pooler = utils.expand_tensor(visn_pooler, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                state[0] = state[0].squeeze(0)
                state[0] = utils.expand_tensor(state[0], beam_size)
                state[0] = state[0].unsqueeze(0)

                p_att_feats_tmp = []
                for p_feat in p_att_feats:
                    p_key, p_value2 = p_feat
                    p_key = utils.expand_tensor(p_key, beam_size)
                    p_value2 = utils.expand_tensor(p_value2, beam_size)
                    p_att_feats_tmp.append((p_key, p_value2))

                p_att_feats = p_att_feats_tmp

        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, self.hparams.bert_answer_max_length))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, self.hparams.bert_answer_max_length))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        self.decoder.clear_buffer()
        return outputs, log_probs


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

        return loss2mean["loss"]

    def validation_step(self, batch, batch_idx):
        # print(f'rank{self.global_rank} in validation step')
        ref = batch['target_ids']
        if self.hparams.beam_size > 1:
            seq, _ = self.decode_beam(batch)
        else:
            seq, _ = self.decode(batch, greedy_decode=self.hparams.greedy_decode)
        return seq, ref

    def validation_epoch_end(self, outputs):
        """
        PL HOOK,每个val epoch end，收集recorder的结果，并且汇总，存储checkpoints
        Args:
            outputs:
        Returns:
        """
        ref, hypo = [], []
        if self.trainer.gpus != 1:
            allgather = self.all_gather(outputs)
            for i in allgather:
                hypo.append(i[0].view(-1, i[0].shape[-1]))
                ref.append(i[1].view(-1, i[1].shape[-1]))
        else:
            for i in outputs:
                hypo.append(i[0])
                ref.append(i[1])

        ref = torch.cat(ref, 0)
        hypo = torch.cat(hypo, 0)

        refs, hypos = {}, {}
        for i, (j, k) in enumerate(zip(ref, hypo)):
            refs[i] = [self.tokenizer.decode(j, skip_special_tokens=True).replace('.', ' .')]
            hypos[i] = [self.tokenizer.decode(k, skip_special_tokens=True).replace('.', ' .')]
        eval_res = self.score(ref=refs, hypo=hypos)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypos, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(refs, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight
        return val_score


    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        normal_optimizer, normal_scheduler = config_optimizer(
            list(self.fc_fusion_layer.parameters()),
            self.hparams.learning_rate, self.hparams.warmup_steps, self.hparams.max_steps,
            name="lr"
        )

        vision_optimizer, vision_scheduler = config_nodecay_optimizer(
            self.vision_model, self.hparams.vision_learning_rate, self.hparams.vision_warmup_steps,
            self.hparams.max_steps,
            name="vision_lr"
        )
        decoder_optimizer, decoder_scheduler = config_decoder_optimizer(
            list(self.decoder.parameters()), self.hparams.decoder_lr, self.hparams.decoder_warmup_steps,
            self.hparams.model_size,
            name='decoder_lr'
        )

        bert_optimizer, bert_scheduler = config_nodecay_optimizer(
            self.bert_model, self.hparams.bert_learning_rate, self.hparams.bert_warmup_steps, self.hparams.max_steps,
            name="bert_lr"
        )

        opts = [normal_optimizer, vision_optimizer, decoder_optimizer, bert_optimizer]
        schedulers = [normal_scheduler, vision_scheduler, decoder_scheduler, bert_scheduler]
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

    @staticmethod
    def swin_pool(hidden_state, masks):
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
        return bert_mean_pooler


    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad()
