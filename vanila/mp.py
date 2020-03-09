import torch
import torch.nn as nn


class BaseMean_10160(nn.Module):
  #output: bs*7*64
  def __init__(self, opt):
    super(BaseMean_10160, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.w1 = nn.Linear(opt.em_dim,opt.em_dim)
    self.drop = nn.Dropout(opt.dropout)

  def forward(self, embeddings):
    u_em = embeddings[:,0,:]
    i_em = embeddings[:,1:,:]
    u_em = torch.unsqueeze(u_em, dim=-2)
    u_em = u_em.expand_as(i_em)
    ui = u_em * i_em
    mu = i_em.mean(dim=1,keepdim=True)
    mu_exp = mu.expand_as(i_em)
    adj = i_em * mu_exp
    adj = self.w1(adj)
    ui = ui + adj
    ui_adj = self.activation_function(ui)
    out_embeddings = self.drop(ui_adj)
    return out_embeddings

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

