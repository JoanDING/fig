import torch
import torch.nn as nn
from torch.nn import functional as F
import attentive_mp as mp


class Model_att(nn.Module):
  def __init__(self, opt):
    super(Model_att, self).__init__()
    em_dim = opt.em_dim
    vocab_size = opt.vocab_size
    self.DEVICE = opt.DEVICE
    self.lr = opt.lr
    self.lr_att = opt.lr_att
    self.em_layer = nn.Embedding(vocab_size, em_dim)
    self.params_em = self.em_layer.parameters()
    self.params = list(self.params_em)

    self.att_layer = mp.att0_16(opt)
    self.scorer = mp.Scorers_w_id(opt)
    self.params_mp = self.att_layer.att_layer.mp_w.parameters()

    self.params += list(self.params_mp)
    self.params_att = list(self.att_layer.att_layer.att_a.parameters())
    self.params_sco = self.scorer.parameters()
    self.params += list(self.params_sco)

    self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=opt.weight_decay)
    self.optimizer_att = torch.optim.Adam(self.params_att, lr=self.lr_att, weight_decay=opt.weight_decay_att)

  def get_output(self, inds, mask):
    inds = inds.to(self.DEVICE)
    mask = mask.to(self.DEVICE)
    em = self.em_layer(inds) # input: uid,iid,att,bs*8 || output: uid_emb,iid_emb,att_emb bs*8*64
    em_update = self.att_layer(em) # input: bs*7*64 || output: bs*7*64
    scores = self.scorer(em_update, mask[:,1:]) # input: bs*7*64 || output:bs*1
    return scores

  def get_loss(self, pos, neg, mask_pos, mask_neg):
    pos_sc = self.get_output(pos, mask_pos)
    neg_sc = self.get_output(neg, mask_neg)
    loss = F.softplus(neg_sc-pos_sc)
    loss = loss.mean()
    # print('loss shape: {}'.format(loss.size()))
    self.loss = loss
    return loss


  def train_step(self, pos, neg, mask_pos, mask_neg):
    pos = pos.to(self.DEVICE)
    neg = neg.to(self.DEVICE)
    mask_pos = mask_pos.to(self.DEVICE)
    mask_neg = mask_neg.to(self.DEVICE)
    loss = self.get_loss(pos, neg, mask_pos, mask_neg)
    self.optimizer.zero_grad()
    self.optimizer_att.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.optimizer_att.step()

