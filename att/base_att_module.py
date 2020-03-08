import torch
import torch.nn as nn
from torch.nn import functional as F

class Att_trans_sum(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(Att_trans_sum,self).__init__()
        self.att_W = nn.Linear(em_dim,em_dim)
        self.att_a = nn.Linear(em_dim,1)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self,embeddings):
        query = torch.unsqueeze(embeddings,dim=-2)
        key = torch.unsqueeze(embeddings,dim=-3)
        query = query.expand(query.size()[0], query.size()[1], query.size()[1], query.size()[-1])
        key = key.expand(key.size()[0], key.size()[2], key.size()[2], key.size()[-1])
        self.wq = self.att_W(query)
        self.wk = self.att_W(key)
        qk = self.wq + self.wk # 256,6,6,64
        qk = self.att_a(qk) #256,6,6,1
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        qk = torch.squeeze(qk, dim=-1) #256,6,6
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        value = query * key
        return alphas,value

# 2 -----------------------------------------------------------

class Att_cat_norm_inte(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(Att_cat_norm_inte,self).__init__()
        self.att_a = nn.Linear(3*em_dim,1)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(em_dim)

    def forward(self,embeddings):
        query = torch.unsqueeze(embeddings,dim=-2)
        key = torch.unsqueeze(embeddings,dim=-3)
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        value = query * key
        self.wq = self.layer_norm(query)
        self.wk = self.layer_norm(key)
        self.wv = self.layer_norm(value)
        qk = torch.cat((self.wq,self.wk,self.wv),dim=-1) # 256,6,6,128
        qk = self.att_a(qk) #256,6,6,1
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        qk = torch.squeeze(qk,dim=-1) #256,6,6
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        return alphas,value

class Att_cat_norm(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(Att_cat_norm,self).__init__()
        self.att_a = nn.Linear(2*em_dim,1)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(em_dim)

    def forward(self,embeddings):
        query = torch.unsqueeze(embeddings,dim=-2)
        key = torch.unsqueeze(embeddings,dim=-3)
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        self.wq = self.layer_norm(query)
        self.wk = self.layer_norm(key)
        qk = torch.cat((self.wq,self.wk),dim=-1) # 256,6,6,128
        qk = self.att_a(qk) #256,6,6,1
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        qk = torch.squeeze(qk,dim=-1) #256,6,6
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        value = query * key
        return alphas,value

class Att_multiply(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(Att_multiply,self).__init__()
        self.att_a = nn.Linear(em_dim,1)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self,embeddings):
        query = torch.unsqueeze(embeddings,dim=-2)
        key = torch.unsqueeze(embeddings,dim=-3)
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        self.wq = query
        self.wk = key
        qk = self.wq*self.wk # 256,6,6,64
        qk = self.att_a(qk) #256,6,6,1
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        qk = torch.squeeze(qk,dim=-1) #256,6,6
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        value = query * key
        return alphas,value

class AFM(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(AFM,self).__init__()
        self.att_a = nn.Linear(em_dim,1)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.w = nn.Linear(em_dim,em_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self,embeddings):
        query = torch.unsqueeze(embeddings,dim=-2)
        key = torch.unsqueeze(embeddings,dim=-3)
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        value = query * key
        qk = self.w(value)
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        qk = self.att_a(qk) #256,6,6,1
        qk = torch.squeeze(qk,dim=-1)
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        return alphas,value


class Att_alpha(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(Att_alpha,self).__init__()
        self.att_a = nn.Linear(2* em_dim,1)
        self.mp_w = nn.Linear(em_dim,em_dim)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self,i_em):
        query = torch.unsqueeze(i_em,dim=-2)
        key = torch.unsqueeze(i_em,dim=-3)
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        self.wq = query
        self.wk = key
        qk = torch.cat((self.wq,self.wk),dim=-1) # 256,6,6,128
        qk = self.att_a(qk) #256,6,6,1
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        qk = torch.squeeze(qk, dim=-1) #256,6,6
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        value = query * key
        return alphas,value


class Att_score(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(Att_score,self).__init__()
        self.mp_w = nn.Linear(em_dim,em_dim)
        self.att_a = nn.Linear(2* em_dim,1)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self,i_em):
        query = torch.unsqueeze(i_em,dim=-2)
        key = torch.unsqueeze(i_em,dim=-3)
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        self.wq = query
        self.wk = key
        qk = torch.cat((self.wq,self.wk),dim=-1) # 256,6,6,128
        qk = self.att_a(qk) #256,6,6,1
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        #qk = self.att_a(qk) #256,6,6,1
        qk = torch.squeeze(qk, dim=-1) #256,6,6
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        return alphas

class Att_add_mp_best(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(Att_add_mp_best,self).__init__()
        self.mp_w = nn.Linear(em_dim,em_dim)
        self.att_a = nn.Linear(2* em_dim,1)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self,i_em):
        query = torch.unsqueeze(i_em,dim=-2)
        key = torch.unsqueeze(i_em,dim=-3)
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        self.wq = query
        self.wk = key
        qk = torch.cat((self.wq,self.wk),dim=-1) # 256,6,6,128
        qk = self.att_a(qk) #256,6,6,1
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        #qk = self.att_a(qk) #256,6,6,1
        qk = torch.squeeze(qk, dim=-1) #256,6,6
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        value = query * key
        return alphas, value


class Att_add_mp_norm(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(Att_add_mp_norm,self).__init__()
        self.att_a = nn.Linear(2* em_dim,1)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(em_dim)

    def forward(self,i_em):
        i_em_norm = self.layer_norm(i_em)
        query = torch.unsqueeze(i_em_norm,dim=-2)
        key = torch.unsqueeze(i_em_norm,dim=-3)
        query_ = torch.unsqueeze(i_em,dim=-2)
        key_ = torch.unsqueeze(i_em,dim=-3)

        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        self.wq = query
        self.wk = key
        qk = torch.cat((self.wq,self.wk),dim=-1) # 256,6,6,128
        qk = self.att_a(qk) #256,6,6,1
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        #qk = self.att_a(qk) #256,6,6,1
        qk = torch.squeeze(qk, dim=-1) #256,6,6
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        value = query_ * key_
        return alphas,value

class Att_add_mp(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(Att_add_mp,self).__init__()
        self.att_a = nn.Linear(2* em_dim,1)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self,i_em):
        query = torch.unsqueeze(i_em,dim=-2)
        key = torch.unsqueeze(i_em,dim=-3)
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        self.wq = query
        self.wk = key
        qk = torch.cat((self.wq,self.wk),dim=-1) # 256,6,6,128
        qk = self.att_a(qk) #256,6,6,1
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        #qk = self.att_a(qk) #256,6,6,1
        qk = torch.squeeze(qk, dim=-1) #256,6,6
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        value = query * key
        return alphas,value

class Att_cat(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(Att_cat,self).__init__()
        self.att_a = nn.Linear(2*em_dim,1)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self,embeddings):
        query = torch.unsqueeze(embeddings,dim=-2)
        key = torch.unsqueeze(embeddings,dim=-3)
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        self.wq = query
        self.wk = key
        qk = torch.cat((self.wq,self.wk),dim=-1) # 256,6,6,128
        qk = self.att_a(qk) #256,6,6,1
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        #qk = self.att_a(qk) #256,6,6,1
        qk = torch.squeeze(qk,dim=-1) #256,6,6
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        value = query * key
        return alphas,value

# 3 -----------------------------------------------------------
class Att_trans_cat(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(Att_trans_cat,self).__init__()
        self.att_W = nn.Linear(em_dim,em_dim)
        self.att_a = nn.Linear(2*em_dim,1)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self,embeddings):
        query = torch.unsqueeze(embeddings,dim=-2)
        key = torch.unsqueeze(embeddings,dim=-3)
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        self.wq = self.att_W(query)
        self.wk = self.att_W(key)
        qk = torch.cat((self.wq,self.wk),dim=-1) # 256,6,6,128
        qk = self.att_a(qk) #256,6,6,1
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        qk = torch.squeeze(qk,dim=-1) #256,6,6
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        value = query * key
        return alphas,value

# 4 -----------------------------------------------------------
class Att_trans_cat_inte(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(Att_trans_cat_inte,self).__init__()
        self.att_W = nn.Linear(em_dim,em_dim)
        self.att_a = nn.Linear(3*em_dim,1)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self,embeddings):
        query = torch.unsqueeze(embeddings,dim=-2)
        key = torch.unsqueeze(embeddings,dim=-3)
        value = query * key
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        self.wq = self.att_W(query)
        self.wk = self.att_W(key)
        self.wv = self.att_W(value)
        qk = torch.cat((self.wq,self.wk,self.wv),dim=-1) # 256,6,6,128
        qk = self.att_a(qk) #256,6,6,1
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        qk = torch.squeeze(qk,dim=-1) #256,6,6
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        return alphas,value

# 5 -----------------------------------------------------------

class Att_cat_withid(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(Att_cat_withid,self).__init__()
        self.att_a = nn.Linear(3*em_dim,1)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self,ua,iid):
        # key: 256,6,6,64; query:256,6,6,64; iid: 256,6,6,64
        query = torch.unsqueeze(ua,dim=-2)
        key = torch.unsqueeze(ua,dim=-3)
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        iid = iid.expand_as(key)
        self.wq = query
        self.wk = key
        self.wiid = iid
        qk = torch.cat((self.wq,self.wk,self.wiid),dim=-1) # 256,6,6,192
        qk = self.att_a(qk) #256,6,6,1
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        qk = torch.squeeze(qk,dim=-1) #256,6,6
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        value = query * key
        alphas = alphas.expand_as(value) #256,6,6,64
        return alphas,value

class Att_trans_cat_withid(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(Att_trans_cat_withid,self).__init__()
        self.att_W = nn.Linear(em_dim,em_dim)
        self.att_a = nn.Linear(3*em_dim,1)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self,ua,iid):
        # key: 256,6,6,64; query:256,6,6,64; iid: 256,6,6,64
        query = torch.unsqueeze(ua,dim=-2)
        key = torch.unsqueeze(ua,dim=-3)
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        iid = iid.expand_as(key)
        self.wq = self.att_W(query)
        self.wk = self.att_W(key)
        self.wiid = self.att_W(iid)
        qk = torch.cat((self.wq,self.wk,self.wiid),dim=-1) # 256,6,6,192
        qk = self.att_a(qk) #256,6,6,1
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        qk = torch.squeeze(qk,dim=-1) #256,6,6
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        value = query * key
        alphas = alphas.expand_as(value) #256,6,6,64
        return alphas,value

# 6 -----------------------------------------------------------
class Att_cat_withid_norm(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(Att_cat_withid_norm,self).__init__()
        self.att_a = nn.Linear(3*em_dim,1)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(em_dim)

    def forward(self,ua,iid):
        # key: 256,6,6,64; query:256,6,6,64; iid: 256,6,6,64
        query = torch.unsqueeze(ua,dim=-2)
        key = torch.unsqueeze(ua,dim=-3)
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        value = query * key
        iid = iid.expand_as(key)
        self.wq = self.layer_norm(query)
        self.wk = self.layer_norm(key)
        self.wiid = self.layer_norm(iid)
        qk = torch.cat((self.wq,self.wk,self.wiid),dim=-1) # 256,6,6,192
        qk = self.att_a(qk) #256,6,6,1
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        qk = torch.squeeze(qk,dim=-1) #256,6,6
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        value = query * key
        alphas = alphas.expand_as(value) #256,6,6,64
        return alphas,value

class Att_cat_withid_inte_norm(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(Att_cat_withid_inte_norm,self).__init__()
        self.att_a = nn.Linear(4*em_dim,1)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(em_dim)

    def forward(self,ua,iid):
        # key: 256,6,6,64; query:256,6,6,64; iid: 256,6,6,64
        query = torch.unsqueeze(ua,dim=-2)
        key = torch.unsqueeze(ua,dim=-3)
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        value = query * key
        iid = iid.expand_as(key)
        self.wq = self.layer_norm(query)
        self.wk = self.layer_norm(key)
        self.wv = self.layer_norm(value)
        self.wiid = self.layer_norm(iid)
        qk = torch.cat((self.wq,self.wk,self.wv,self.wiid),dim=-1) # 256,6,6,192
        qk = self.att_a(qk) #256,6,6,1
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        qk = torch.squeeze(qk,dim=-1) #256,6,6
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        value = query * key
        alphas = alphas.expand_as(value) #256,6,6,64
        return alphas,value

class Att_trans_cat_withid_inte(nn.Module):
    def __init__(self,activation_fun,dropout,em_dim):
        super(Att_trans_cat_withid_inte,self).__init__()
        self.att_W = nn.Linear(em_dim,em_dim)
        self.att_a = nn.Linear(4*em_dim,1)
        if activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self,ua,iid):
        # key: 256,6,6,64; query:256,6,6,64; iid: 256,6,6,64
        query = torch.unsqueeze(ua,dim=-2)
        key = torch.unsqueeze(ua,dim=-3)
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        value = query * key
        iid = iid.expand_as(key)
        self.wq = self.att_W(query)
        self.wk = self.att_W(key)
        self.wv = self.att_W(value)
        self.wiid = self.att_W(iid)
        qk = torch.cat((self.wq,self.wk,self.wv,self.wiid),dim=-1) # 256,6,6,192
        qk = self.att_a(qk) #256,6,6,1
        qk = self.activation_function(qk)
        qk = self.drop(qk)
        qk = torch.squeeze(qk,dim=-1) #256,6,6
        alphas = F.softmax(qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(alphas,dim=-1)
        value = query * key
        alphas = alphas.expand_as(value) #256,6,6,64
        return alphas,value

