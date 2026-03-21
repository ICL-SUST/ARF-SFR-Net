import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models
import numpy as np

from . import ARFSFR_Conv_4
from . import ARFSFR_ResNet
from .backbones import Conv_4,ResNet
from .backbones.SRM import SRM
from .backbones.MRM import MRM

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '3,2,1,0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ARFSFR(nn.Module):
    
    def __init__(self,way=None,shots=None,resnet=False):
        
        super().__init__()
        

        self.resolution = 5*5
        if resnet:
            self.num_channel = 640
            self.feature_extractor = ARFSFR_ResNet.dctBackbone()
            self.dim = self.num_channel*5*5
            self.feature_extractor = nn.DataParallel(self.feature_extractor, device_ids=[0,1,2])
            
        else:
            self.num_channel = 64
            self.feature_extractor = ARFSFR_Conv_4.dctBackbone(self.num_channel)
            self.dim = self.num_channel*5*5
            self.feature_extractor = nn.DataParallel(self.feature_extractor, device_ids=[0,1,2])

        self.fsrm = SRM(
                sequence_length=self.resolution,
                embedding_dim=self.num_channel,
                num_layers=1,
                num_heads=1,
                mlp_dropout_rate=0.,
                attention_dropout=0.,
                positional_embedding='sine')

        self.fmrm = MRM(hidden_size=self.num_channel, inner_size=self.num_channel, num_patch=self.resolution, drop_prob=0.1)

        self.shots = shots
        self.way = way
        self.resnet = resnet

        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)

        self.w1 = nn.Parameter(torch.FloatTensor([0.5]),requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor([0.5]),requires_grad=True)

        n = 9
        if n > 1:
            self.feature_extractor = nn.DataParallel(self.feature_extractor, device_ids=[0,1,2])

    def get_feature_vector(self,inp, epoch=None, hw_range=None):

        batch_size = inp.size(0)
        feature_map = self.feature_extractor.module(inp, epoch=epoch, hw_range=hw_range)
        # print(feature_map.shape)
        feature_map = self.fsrm(feature_map).transpose(1, 2).view(batch_size, self.num_channel, 5, 5)

        return feature_map
    

    def get_neg_l2_dist(self,inp,way,shot,query_shot, epoch=None, hw_range=None):
        # 距离度量
        feature_map = self.get_feature_vector(inp, epoch=epoch, hw_range=hw_range)
        support = feature_map[:way*shot].view(way, shot, *feature_map.size()[1:]).permute(0, 2, 1, 3, 4).contiguous()
        query = feature_map[way*shot:] 
        
        sq_similarity, qs_similarity = self.fmrm(support, query)

        l2_dist = self.w1*sq_similarity + self.w2*qs_similarity
        
        return l2_dist


    
    def meta_test(self,inp,way,shot,query_shot, epoch, hw_range):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                        way=way,
                                        shot=shot,
                                        query_shot=query_shot,
                                        epoch=epoch,
                                        hw_range=hw_range)

        _,max_index = torch.max(neg_l2_dist,1)

        return max_index


    def forward(self,inp, epoch=None, hw_range=None):

        logits = self.get_neg_l2_dist(inp=inp,
                                        way=self.way,
                                        shot=self.shots[0],
                                        query_shot=self.shots[1],
                                        epoch=epoch,
                                        hw_range=hw_range)
        logits = logits/self.dim*self.scale

        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction