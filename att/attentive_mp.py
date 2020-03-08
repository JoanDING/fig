import torch
import torch.nn as nn
from base_att_module import *


class att0_16(nn.Module):
    def __init__(self,opt):
        super(att0_16, self).__init__()
        self.att_layer = Att_add_mp_best(opt.activation_fun, opt.dropout, opt.em_dim)
        self.mp_w = nn.Linear(opt.em_dim, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        u_em = embeddings[:, 0, :]
        i_em = embeddings[:, 1:, :]
        u_em = torch.unsqueeze(u_em, dim=-2)
        u_em = u_em.expand_as(i_em)
        ui = u_em * i_em
        self.alphas, value = self.att_layer(i_em)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        att_value = self.mp_w(att_value)
        ui_att_adj = ui + att_value
        em_out = self.activation_function(ui_att_adj)
        em_out = self.drop(em_out)
        return em_out


class attentive_scorers_w_id(nn.Module):
  def  __init__(self, opt):
    super(attentive_scorers_w_id, self).__init__()
    em_dim = opt.em_dim
    num_node = opt.num_node
    self.DEVICE = opt.DEVICE
    self.share_scorer = opt.share_scorer
    self.classme = opt.classme
    self.att_layer = Att_score(opt.activation_fun, opt.dropout, opt.em_dim)
    if self.share_scorer:
      self.scorers = nn.Linear(em_dim,1)
      print('{} scorers for embeddings scoring'.format(1))
    else:
      self.scorers = nn.ModuleList([nn.Linear(em_dim,1) for i in range(num_node)])
      print('{} scorers for embeddings scoring'.format(len(self.scorers)))
    self.activation = nn.Sigmoid()
    if self.classme:
      self.cls_layer = nn.Linear(num_node, 1)

  def forward(self, embeddings, mask):
    bs, num_node, em_dim = embeddings.size()

    prob = torch.zeros([bs, num_node],device=self.DEVICE)
    for i in range(num_node):
      sc = self.scorers[i]
      em_i = embeddings[:,i,:]
      prob_i = sc(em_i)
      prob_i = torch.squeeze(prob_i,dim=1)
      prob_i = self.activation(prob_i)
      prob[:, i] = prob_i
    prob = prob * mask # add for masking
    alpha = self.att_layer(embeddings)
    prob = prob * alpha
    prob = prob.sum(dim=1)/mask.sum(dim=1) # mean-->sum, for masking
    return prob

class Scorers_w_id(nn.Module):
  def  __init__(self, opt):
    super(Scorers_w_id, self).__init__()
    em_dim = opt.em_dim
    num_node = opt.num_node
    self.DEVICE = opt.DEVICE
    self.share_scorer = opt.share_scorer
    self.classme = opt.classme
    if self.share_scorer:
      self.scorers = nn.Linear(em_dim,1)
      print('{} scorers for embeddings scoring'.format(1))
    else:
      self.scorers = nn.ModuleList([nn.Linear(em_dim,1) for i in range(num_node)])
      print('{} scorers for embeddings scoring'.format(len(self.scorers)))
    self.activation = nn.Sigmoid()
    if self.classme:
      self.cls_layer = nn.Linear(num_node, 1)

  def forward(self, embeddings, mask):
    bs, num_node, em_dim = embeddings.size()

    prob = torch.zeros([bs, num_node],device=self.DEVICE)
    for i in range(num_node):
      sc = self.scorers[i]
      em_i = embeddings[:,i,:]
      prob_i = sc(em_i)
      prob_i = torch.squeeze(prob_i,dim=1)
      prob_i = self.activation(prob_i)
      prob[:, i] = prob_i
    prob = prob * mask # add for masking
    if self.classme:
      prob = self.cls_layer(prob)
      prob = torch.squeeze(prob,dim=1)
    else:
      prob = prob.sum(dim=1)/mask.sum(dim=1) # mean-->sum, for masking
    return prob


