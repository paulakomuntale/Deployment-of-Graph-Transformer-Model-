# app/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.attn = nn.MultiheadAttention(out_features, num_heads, dropout=dropout)
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj=None):
        # x: (num_nodes, batch, features)
        x_fc = self.fc1(x)
        attn_output, _ = self.attn(x_fc, x_fc, x_fc)
        x = self.norm1(x_fc + self.dropout(attn_output))
        out = self.fc2(F.relu(x))
        x = self.norm2(x + self.dropout(out))
        return x


class GraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, num_heads=4):
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList([
            GraphTransformerLayer(input_dim if i == 0 else hidden_dim, hidden_dim, num_heads)
            for i in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj=None):
        # Convert to (num_nodes, batch, features) format for multihead attention
        if x.dim() == 2:
            x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, adj)
        x = x.mean(dim=0)  # Global mean pooling
        return self.fc_out(x)
