import torch
import torch.nn as nn
import re

class MultiheadCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_queries=256, num_heads=8):
        super(MultiheadCrossAttention, self).__init__()
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_size))
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size = x.size(1)
        query = self.query_tokens.unsqueeze(1).expand(-1, batch_size, -1)

        # Project key and value
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Reshape for multi-head attention
        query = query.view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        key = key.view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        value = value.view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled dot-product attention
        attn_weights = torch.bmm(query, key.transpose(1, 2)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.bmm(attn_weights, value)

        # Reshape back to original
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, batch_size, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.permute(1, 2, 0)  # 调整维度以适应池化层
        attn_output = self.pooling(attn_output).squeeze(-1)

        return attn_output

def build_sc_alignment(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':

        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type=='resample':
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        cross_attention_layer = MultiheadCrossAttention(config.hidden_size)
        modules.append(cross_attention_layer)
        return nn.Sequential(*modules)



    raise ValueError(f'Unknown projector type: {projector_type}')