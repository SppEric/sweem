import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepOmixNet(nn.Module):
    def __init__(self, In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes,Pathway_Mask):
        super(DeepOmixNet, self).__init__()
        self.embedding = nn.Linear(In_Nodes, Hidden_Nodes)
        self.self_attention = SelfAttention(Hidden_Nodes)
        self.dense1 = nn.Linear(Hidden_Nodes, 32)
        self.dense2 = nn.Linear(32, 1)
        self.dense3 = nn.Linear(2, 1)

    def forward(self, x_1, x_2):
        embedded = self.embedding(x_1)
        self_attention = self.self_attention(embedded)
        dense1 = self.dense1(self_attention)
        act1 = F.relu(dense1)
        dense2 = self.dense2(act1)
        cat = torch.cat((dense2, x_2), dim=1)
        out = self.dense3(cat)
        risk_score = torch.sigmoid(out)
        return risk_score

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
    # TODO multi head attention
    
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2):
        super(SelfAttentionModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim_1)
        self.self_attention = SelfAttention(hidden_dim_1)
        self.dense1 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.dense2 = nn.Linear(hidden_dim_2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        self_attention = self.self_attention(embedded)
        dense1 = self.dense1(self_attention)
        act1 = F.relu(dense1)
        dense2 = self.dense2(act1)
        risk_score = torch.sigmoid(dense2)
        return risk_score

class CrossAttentionModel(nn.Module):
    def __init__(self, mutation_dim, expression_dim, methylation_dim, cna_dim, hidden_dim):
        super(CrossAttentionModel, self).__init__()
        self.mutation_embedding = nn.Linear(mutation_dim, hidden_dim)
        self.expression_embedding = nn.Linear(expression_dim, hidden_dim)
        self.methylation_embedding = nn.Linear(methylation_dim, hidden_dim)
        self.cna_embedding = nn.Linear(cna_dim, hidden_dim)
        self.self_attention = SelfAttention(hidden_dim)
        self.cross_attention = CrossAttention(hidden_dim)
        self.months = nn.Linear(hidden_dim, 1)
        self.event = nn.Linear(hidden_dim, 1)
    
    def forward(self, mutation, expression, methylation, cna):
        # mutation is binary data
        mutation_embedded = self.mutation_embedding(mutation)
        expression_embedded = self.expression_embedding(expression)
        methylation_embedded = self.methylation_embedding(methylation)
        cna_embedded = self.cna_embedding(cna)
        
        # TODO
        mutation_attention = self.self_attention(mutation_embedded)
        expression_attention = self.self_attention(expression_embedded)
        methylation_attention = self.self_attention(methylation_embedded)
        cna_attention = self.self_attention(cna_embedded)