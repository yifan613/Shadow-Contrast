import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    
    def __init__(self, input_channels, condition_channels, num_heads=8):
        super(CrossAttention, self).__init__()
        assert input_channels % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = input_channels // num_heads
        self.scale = self.head_dim ** 0.5

        self.query_proj_conv = nn.Conv2d(input_channels, input_channels, 1)
        self.key_proj_conv = nn.Conv2d(condition_channels, input_channels, 1)
        self.value_proj_conv = nn.Conv2d(condition_channels, input_channels, 1)
        
        self.query_proj_mlp = nn.Linear(input_channels, input_channels, bias = False)
        self.key_proj_mlp = nn.Linear(condition_channels, input_channels, bias = False)
        self.value_proj_mlp = nn.Linear(condition_channels, input_channels, bias = False)

    def forward(self, x, condition):
        B, _, H, W = x.shape
        _, _, h, w = condition.shape       
        if h != H or w != W:
            condition = F.interpolate(condition, size=(H, W), mode='bilinear', align_corners=False)
        queries = self.query_proj_conv(x).view(B, self.num_heads, self.head_dim, H * W)\
            + self.query_proj_mlp(x.flatten(start_dim=2).transpose(1, 2)).view(B, self.num_heads, self.head_dim, H * W)
        keys = self.key_proj_conv(condition).view(B, self.num_heads, self.head_dim, H * W)\
            + self.key_proj_mlp(condition.flatten(start_dim=2).transpose(1, 2)).view(B, self.num_heads, self.head_dim, H * W)
        values = self.value_proj_conv(condition).view(B, self.num_heads, self.head_dim, H * W)\
            + self.value_proj_mlp(condition.flatten(start_dim=2).transpose(1, 2)).view(B, self.num_heads, self.head_dim, H * W)
        queries = queries.permute(0, 1, 3, 2)   # [2, 8, 262144, 4]
        keys = keys.permute(0, 1, 3, 2)
        values = values.permute(0, 1, 3, 2)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        attention = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention, values).permute(0, 1, 3, 2).contiguous()
        out = out.view(B, -1, H, W)  
        return out
