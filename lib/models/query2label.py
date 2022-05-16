# --------------------------------------------------------
# Quert2Label
# Written by Shilong Liu
# --------------------------------------------------------

from audioop import bias
import imp
from multiprocessing import pool
import os, sys
from re import S
import os.path as osp
from turtle import back, forward

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import math

from models.backbone import build_backbone
from models.transformer import build_transformer
from models.vit import build_vit, build_transformer_encoder, BatchNorm
from utils.misc import clean_state_dict

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class SEModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction):
        super().__init__()
        if in_channels==out_channels:
            self.res = nn.Identity()
        else:
            self.res = nn.Conv2d(in_channels, out_channels, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = self._round_width(in_channels, reduction)
        self.fc1 = nn.Conv2d(
            in_channels, self.bottleneck, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(
            self.bottleneck, out_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        # self.weight_init()

    @staticmethod
    def _round_width(width, multiplier, min_width=8, divisor=8):
        width *= multiplier
        min_width = min_width or divisor
        width_out = max(min_width,
                        int(width + divisor / 2) // divisor * divisor)
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def forward(self, x):
        module_input = self.res(x)
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class TopKMaxPooling(nn.Module):
    def __init__(self, kmax=1.0):
        super(TopKMaxPooling, self).__init__()
        self.kmax = kmax

    @staticmethod
    def get_positive_k(k, n):
        if k <= 0:
            return 0
        elif k < 1:
            return round(k * n)
        elif k > n:
            return int(n)
        else:
            return int(k)

    def forward(self, input):
        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)
        n = h * w  # number of regions
        kmax = self.get_positive_k(self.kmax, n)
        sorted, indices = torch.sort(input.view(batch_size, num_channels, n), dim=2, descending=True)
        region_max = sorted.narrow(2, 0, kmax)
        output = region_max.sum(2).div_(kmax)
        return output.view(batch_size, num_channels)

    def __repr__(self):
        return self.__class__.__name__ + ' (kmax=' + str(self.kmax) + ')'

class GraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GraphConvolution, self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.weight = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, adj, nodes):
        nodes = torch.matmul(nodes, adj)
        nodes = self.relu(nodes)
        nodes = self.weight(nodes)
        nodes = self.relu(nodes)
        return nodes

class Qeruy2Label(nn.Module):
    def __init__(self, backbone, transfomer, num_class):
        """[summary]
    
        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes. (80 for MSCOCO).
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class

        # assert not (self.ada_fc and self.emb_fc), "ada_fc and emb_fc cannot be True at the same time."
        
        hidden_dim = transfomer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)


    def forward(self, input):
        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]
        # import ipdb; ipdb.set_trace()

        query_input = self.query_embed.weight
        hs = self.transformer(self.input_proj(src), query_input, pos)[0] # B,K,d
        out = self.fc(hs[-1])
        # import ipdb; ipdb.set_trace()
        return out

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(), self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))


class LRFormer(nn.Module):
    def __init__(self, 
                 backbone, 
                 vit, 
                #  feat_trans
        ):
        super().__init__()
        self.num_classes = 80
        self.backbone = backbone
        self.vit = vit
        self.base_cls = nn.Conv2d(2048, 80, 1, bias=False)
        self.conv1 = nn.Conv2d(2048, 1024, 1)
        self.bn = BatchNorm(1024)
        self.kmp = TopKMaxPooling(kmax=0.1)
        # self.conv_node = nn.Conv1d(80, 80, 1)
        # self.conv_weight = nn.Conv1d(1024, 1024, 1)
        # self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        # self.lr_cls = nn.Conv1d(1024, 80, 1)
        # self.bn = nn.BatchNorm2d(2048)
        self.lr_cls = GroupWiseLinear(80, 1024)
        self.weight_init()
    
    def weight_init(self):
        for p in self.vit.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # for p in self.feat_trans.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)


    def label_embed(self, x):
        mask = self.base_cls(x)
        # coarse_socre1 = mask.view(mask.size(0), mask.size(1), -1)
        # coarse_socre1 = F.adaptive_max_pool2d(mask, (1, 1))
        # print(coarse_socre1.shape)
        # coarse_socre1 = coarse_socre1.view(coarse_socre1.size(0), -1)
        coarse_socre1 = self.kmp(mask)
        mask = torch.sigmoid(mask)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        
        x = self.conv1(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.transpose(1, 2).contiguous()
        x = torch.matmul(mask, x)
        # x = self.bn(x)

        # node = self.conv_node(x)
        # node = self.conv_weight(node.transpose(1, 2).contiguous())
        # x = x + node.transpose(1, 2).contiguous()
        return x, coarse_socre1
    
    def generate_mask(self, pred):
        b = pred.size(0)
        mask = torch.zeros([b, self.num_classes, self.num_classes]).cuda()
        _, top_pred_ind = torch.topk(pred, 20)
        # _, bottom_pred_ind = torch.topk(pred, 40, largest=False)
        for i in range(b):
            mask[i].index_fill_(0, top_pred_ind[i], 1)
            mask[i].index_fill_(1, top_pred_ind[i], 1)
            # mask[i].index_fill_(0, bottom_pred_ind[i], 0)
            # mask[i].index_fill_(1, bottom_pred_ind[i], 0)
        return mask


    def forward(self, x, mask=None):
        x, _ = self.backbone(x)
        x = x[-1]
        # x = self.bn(x)
        v, score1 = self.label_embed(x)
        # v = v * torch.sigmoid(v)
        v = self.bn(v)
        # mask = None
        mask = self.generate_mask(score1.detach()) #  if not self.training else None
        # mask = mask if not self.training else None
        z = self.vit(v, mask)
        # x = self.last_norm(x)
        score2 = self.lr_cls(z)
        # mask_mat = self.mask_mat.detach()
        # score2 = (score2 * mask_mat).sum(-1)
        cls_score = (score1 + score2) / 2.
        # return score1, score2
        return cls_score
    
    def finetune_paras(self):
        from itertools import chain
        return chain(self.vit.parameters(), 
                    #  self.base_cls.parameters(), 
                     self.conv1.parameters(), 
                    #  self.conv_transform2.parameters(), 
                    #  self.feat_trans.parameters(),
                    #  self.conv2.parameters(), 
                     self.lr_cls.named_parameters(),
                     self.bn.parameters(),
                    #  self.mask_mat,
                    #  self.last_bn.parameters()
                    )

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.backbone.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [
                {'params': self.backbone.parameters(), 'lr': lr*lrp},
                {'params': large_lr_layers, 'lr': lr},
                ]

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))


class LRFormer3D(nn.Module):
    def __init__(self, backbone, vit):
        super().__init__()
        self.num_classes = 80
        self.backbone = backbone
        self.vit = vit
        self.base_cls = nn.Conv3d(2048, 80, 1, bias=False)
        self.conv1 = nn.Conv3d(2048, 1024, 1)
        self.bn = BatchNorm(1024)
        self.fc_cls = GroupWiseLinear(80, 1024)
        # self.se = SEModule(2048, 1024, 1/2)
        self.weight_init()
    
    def weight_init(self):
        for p in self.vit.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def label_embed(self, x):
        # g_f = F.adaptive_max_pool2d(x, (1, 1))
        # coarse_socre = self.base_cls(g_f)
        # coarse_socre = coarse_socre.view(coarse_socre.size(0), -1)
        mask = self.base_cls(x)
        coarse_socre = mask.view(mask.size(0), mask.size(1), -1)
        # coarse_socre = F.adaptive_max_pool2d(mask, (1, 1))
        # coarse_socre = coarse_socre.view(coarse_socre.size(0), -1)
        coarse_socre = coarse_socre.topk(1, dim=-1)[0].mean(dim=-1)
        # coarse_socre = coarse_socre.max(-1)[0]
        mask = torch.sigmoid(mask)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        
        x = self.conv1(x)
        # x = self.se(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.transpose(1, 2).contiguous()
        x = torch.matmul(mask, x)
        return x, coarse_socre
    
    def generate_mask(self, pred):
        b = pred.size(0)
        mask = torch.zeros([b, self.num_classes, self.num_classes])
        _, top_pred_ind = torch.topk(pred, 20)
        for i in range(b):
            for j in top_pred_ind[i]:
                mask[i,j:j+1,:] = 1
                mask[i,:,j:j+1] = 1
        return mask
        # for j in range(self.num_classes):
        #     for k in range(self.num_classes):

        # mask_ind = 

    def forward(self, x):
        x, _ = self.backbone(x)
        # len = 1
        x = x[-1]
        x, score1 = self.label_embed(x)
        mask = self.generate_mask(score1.detach())
        
        # x = x.transpose(1, 2).contiguous()
        x = self.bn(x)
        # x = x.transpose(1, 2).contiguous()

        x = self.vit(x, mask)

        score2 = self.fc_cls(x)
        # return score2
        cls_score = (score1 + score2) / 2.
        return cls_score
    
    def finetune_paras(self):
        from itertools import chain
        return chain(self.vit.parameters(), 
                     self.base_cls.parameters(), 
                     self.conv1.parameters(), 
                     self.fc_cls.parameters(), 
                    #  self.se.parameters(),
                    #  self.input_bn.parameters(), 
                     self.bn.parameters(),
                    #  self.last_bn.parameters()
                    )

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))


def build_q2l(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = Qeruy2Label(
        backbone = backbone,
        transfomer = transformer,
        num_class = args.num_class
    )

    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("set model.input_proj to Indentify!")
    

    return model

def build_LRFormer(args):
    backbone = build_backbone(args)
    vit = build_vit(args.num_class, 1024, 2, 8, mlp_dim=1024*2, in_channels=1024, dim_head=1024//8, dropout=0., emb_dropout=0., pool=None, use_cls_token=False)
    # feat_Trans = build_vit(args.num_class, 512, 1, 4, mlp_dim=1024, in_channels=1024, dim_head=128, dropout=0., emb_dropout=0., pool=None, use_cls_token=False, img_size=448)
    model = LRFormer(
        backbone = backbone,
        vit = vit,
        # feat_trans=feat_Trans
    )
    return model      

def build_LRFormer3D(args):
    backbone = build_backbone(args)
    vit = build_vit(args.num_class, 512, 2, 8, mlp_dim=1024, in_channels=512, dim_head=64, dropout=0., emb_dropout=0.)
    model = LRFormer(
        backbone = backbone,
        vit = vit
    )
    return model   
        
