import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttResNet(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttResNet, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dense = nn.Linear(input_dim + input_dim, input_dim)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.mm(queries, keys.transpose(0,1)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)   
        weighted = torch.mm(attention, values)
        cat = torch.cat((x, weighted), dim=1)
        out = self.dense(cat)
        out = F.relu(out)
        return out

class SWEEMv2(nn.Module):
    def __init__(self, rna_dim, scna_dim, methy_dim):
        super(SWEEMv2, self).__init__()
        self.rna_att = SelfAttResNet(rna_dim)
        self.scna_att = SelfAttResNet(scna_dim)
        self.methyl_att = SelfAttResNet(methy_dim)
        self.cross_att = SelfAttResNet(rna_dim + scna_dim + methy_dim)
        self.dense = nn.Linear(rna_dim + scna_dim + methy_dim + 1, 1)

    def forward(self, rna, scna, methy, event):
        rna_att = self.rna_att(rna)
        scna_att = self.scna_att(scna)
        methy_att = self.methyl_att(methy)
        cat = torch.cat((rna_att, scna_att, methy_att), dim=1)
        # cross_att = self.cross_att(cat)
        cat = torch.cat((cat, event), dim=1)
        out = self.dense(cat)
        out = F.sigmoid(out)
        return out

class SWEEM(nn.Module):
    def __init__(self, rna_dim, scna_dim, methy_dim):
        super(SWEEM, self).__init__()
        self.rna_att = SelfAttResNet(rna_dim)
        self.scna_att = SelfAttResNet(scna_dim)
        self.methyl_att = SelfAttResNet(methy_dim)
        self.cross_att = SelfAttResNet(rna_dim + scna_dim + methy_dim)
        self.dense = nn.Linear(rna_dim + scna_dim + methy_dim + 1, 1)

    def forward(self, rna, scna, methy, event):
        rna_att = self.rna_att(rna)
        scna_att = self.scna_att(scna)
        methy_att = self.methyl_att(methy)
        cat = torch.cat((rna_att, scna_att, methy_att), dim=1)
        # cross_att = self.cross_att(cat)
        cat = torch.cat((cat, event), dim=1)
        out = self.dense(cat)
        out = F.sigmoid(out)
        return out