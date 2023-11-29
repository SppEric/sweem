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
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
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
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttentionModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.self_attention = SelfAttention(hidden_dim)
        self.months = nn.Linear(hidden_dim, 1)
        self.event = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        self_attention = self.self_attention(embedded)
        event = self.event(self_attention)
        event = torch.sigmoid(event)
        months = self.months(self_attention)
        return event, months

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