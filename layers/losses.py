import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchsnooper


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, logits, target):
        logprobs = F.log_softmax(logits, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()


class SimSceLayer(nn.Module):

    def __init__(self, tau=0.05):
        super(SimSceLayer, self).__init__()
        self.tau = tau
        self.big_positive = 1e10

    def forward(self, batch_feat1, batch_feat2, input_mask=None):
        """
        Normally, batch_feat2 comes from argument of batch_feat1, for example re dropout.
        Args:
            batch_feat1: Shape (b, feat_size)
            batch_feat2: Shape (b, feat_size)
            input_mask:  Shape (b,) contains 1 and 0, 1 means should not to be attended
        Returns:
        """
        tau = self.tau
        x1, x2 = batch_feat1, batch_feat2

        batch_feat = torch.cat([x1, x2], dim=0)

        batch_size, feat_size = batch_feat.size()
        half_batch_size = int(batch_size / 2)

        similarity = F.cosine_similarity(batch_feat.unsqueeze(1), batch_feat.unsqueeze(0), dim=2)

        mask = torch.eye(batch_size).to(similarity.device).float()
        logits = similarity / tau - mask * self.big_positive  # batch, batch

        idx = torch.arange(half_batch_size).to(similarity.device)
        y_true = torch.cat([idx + half_batch_size, idx])  # [4, 5, 6 ,7] + [0, 1, 2, 3]

        if input_mask is not None:
            input_mask = torch.cat([input_mask, input_mask], dim=0)  # batch x 2, 1 means value should be ignore
            loss = F.cross_entropy(logits, y_true, weight=(1 - input_mask).float())
        else:
            loss = F.cross_entropy(logits, y_true)

        return loss, logits, y_true


class MatchingLayer(nn.Module):

    def __init__(self, tau=0.05):
        super(MatchingLayer, self).__init__()
        self.tau = tau
        self.eps = 1e-5

    # @torchsnooper.snoop()
    def forward(self, feat1, feat2, mask=None):
        """
        matching feat1 and feat2 based on cosine similarity.
        Args:
            feat1: Shape (b1, feat_size)
            feat2: Shape (b2, feat_size)
            mask: Shape (b1,) contains 1 and 0, 1 means should not to be attended
        Returns:
        """
        tau = self.tau
        b1 = feat1.size(0)
        logits = F.cosine_similarity(x1=feat1.unsqueeze(1), x2=feat2.unsqueeze(0), dim=2) / tau  # b1, b2

        gt = torch.arange(b1, device=feat1.device)
        if mask is not None:
            loss = F.cross_entropy(logits, gt, reduction="none")
            loss *= (1 - mask).to(loss.dtype)
            loss = loss.sum() / ((1 - mask).sum() + self.eps)
        else:
            loss = F.cross_entropy(logits, gt)

        return loss, logits, gt


class MoCoLayer(nn.Module):

    def __init__(self, tau=0.05):
        super(MoCoLayer, self).__init__()
        self.tau = tau
        self.eps = 1e-5

    # @torchsnooper.snoop()
    def forward(self, query, keys, memories, mask=None):
        """
        Args:
            query:
            keys:
            memories:
            mask: Shape (b1,) contains 1 and 0, 1 means should not to be attended
        Returns:
        """
        tau = self.tau

        query = F.normalize(query)
        keys = F.normalize(keys)
        memories = F.normalize(memories)
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = F.cosine_similarity(query, keys).unsqueeze(-1)
        l_neg = F.cosine_similarity(query.unsqueeze(1), memories.unsqueeze(0), dim=2)
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= tau

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query.device)

        if mask is None:
            loss = F.cross_entropy(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels, reduction="none")
            loss *= (1 - mask).to(loss.dtype)
            loss = loss.sum() / ((1 - mask).sum() + self.eps)

        return loss


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        # self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, logit, target_seq, weight=None):
        seq_length = logit.shape[1]
        logit = logit.view(-1, logit.shape[-1])
        target_seq = target_seq.view(-1)
        # loss = self.criterion(logit, target_seq, weight)
        # return loss
        if weight is not None:
            loss = F.cross_entropy(logit, target_seq, reduction='none', ignore_index=0)
            loss = torch.mean((weight.view(-1) * loss)*seq_length)
        else:
            loss = F.cross_entropy(logit, target_seq, reduction='mean', ignore_index=0)
        return loss


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, seq, logP, rewards):
        mask = seq > 0
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
        rewards = rewards.view(-1, 1).expand_as(logP)
        logP = torch.masked_select(logP, mask)
        rewards = torch.masked_select(rewards, mask)
        loss = torch.mean(-logP * rewards)
        return loss


class ConsistencyMse(nn.Module):
    def __init__(self):
        super(ConsistencyMse, self).__init__()

    def forward(self, input_logits, target_logits):
        """Takes softmax on both sides and returns MSE loss
        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
        if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_softmax = F.sigmoid(input_logits)
        target_softmax = F.sigmoid(target_logits)
        num_classes = input_logits.size()[1]
        return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes



class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps, max=1-self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps, max=1-self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                            self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)   
                self.loss *= self.asymmetric_w         
        _loss = - self.loss.sum() / x.size(0)
        _loss = _loss / y.size(1) * 1000

        return _loss


if __name__ == "__main__":
    from torch.nn import functional
    loss = AsymmetricLoss()
    x = torch.rand(64, 100)
    y =torch.randint(0, 99, (64,))
    y = functional.one_hot(y, num_classes=100)
    out = loss(x, y)