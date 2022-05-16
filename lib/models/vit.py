from turtle import forward
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from lib.models.swin_transformer import build_swin_transformer


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class BatchNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = self.norm(x)
        x = x.transpose(1, 2).contiguous()
        return x

class PreBatchNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = BatchNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = mask.cuda()
            mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
            dots = dots.float().masked_fill(mask == 0, -1e9)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.depth = depth
        # self.mask_config = [1, 1]
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = 0.)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            # mask = mask if self.mask_config[i]==1 else None
            x = attn(x, mask=mask) + x
            x = ff(x) + x
            
        return x


class ViT(nn.Module):
    def __init__(self, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., img_size=None, use_cls_token = False):
        super().__init__()        
        self.dropout = nn.Dropout(emb_dropout)
        if channels!=dim:
            self.to_patch_embedding = nn.Linear(channels, dim)
        else:
            self.to_patch_embedding = nn.Identity()

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.use_cls_token = use_cls_token
        self.pool = pool
        if self.use_cls_token:
            # self.pos_embedding = nn.Parameter(torch.randn(1, (img_size//32)**2+1, dim))
            self.pos_embedding = nn.Parameter(torch.randn(1, num_classes+1, dim))
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
                # self.to_latent = nn.Identity()
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )
        elif img_size != None:
            self.pos_embedding = nn.Parameter(torch.randn(1, (img_size//16)**2, dim))
            self.cls_token = None
            self.mlp_head = nn.Identity()
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_classes, dim))
            self.cls_token = None
            self.mlp_head = nn.Identity()

    def forward(self, x, mask=None):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        if self.use_cls_token and self.pool != None:
            cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :n+1]
        else:
            x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x, mask)
        if self.use_cls_token and self.pool != None:
            # cls_list = torch.stack(cls_list, dim=1)
            # cls_list, _ = cls_list.max(dim=1)
            x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
            # x = self.to_latent(x)
                
            return self.mlp_head(x)

        return x

def build_vit(
        num_classes, 
        dim,
        depth,
        heads, 
        mlp_dim, 
        in_channels = 3, 
        dim_head = 64, 
        dropout = 0., 
        emb_dropout = 0.,
        img_size = 448,
        use_cls_token = False,
        pool='cls'):
    return ViT(
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=in_channels,
        dim_head=dim_head,
        dropout=dropout,
        emb_dropout=emb_dropout,
        img_size=img_size,
        use_cls_token=use_cls_token,
        pool=pool
    )

def build_transformer_encoder(dim, depth, heads, dim_head, mlp_dim, dropout):
    return Transformer(
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        mlp_dim=mlp_dim,
        dropout=dropout
    )
