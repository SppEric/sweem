import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.mm(queries, keys.transpose(0,1)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)   
        weighted = torch.mm(attention, values)
        return weighted

class CrossAttention(nn.Module):
    def __init__(self, input_dim):
        super(CrossAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x, y):
        queries = self.query(x)
        keys = self.key(y)
        values = self.value(y)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted

class SelfAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim_1 = 1024, hidden_dim_2 = 32):
        super(SelfAttentionModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim_1)
        self.self_attention = SelfAttention(hidden_dim_1)
        self.dense1 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.dense2 = nn.Linear(hidden_dim_2, 1)
        self.dense3 = nn.Linear(2, 1)

    def forward(self, x, x2):
        embedded = self.embedding(x)
        # self_attention = self.self_attention(embedded)
        dense1 = self.dense1(embedded)
        act1 = F.relu(dense1)
        dense2 = self.dense2(act1)
        cat = torch.cat((dense2, x2), dim=1)
        dense3 = self.dense3(cat)
        risk_score = torch.sigmoid(dense3)
        return risk_score