import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.gcn import TreeGCN

from math import ceil

class Discriminator(nn.Module):
    def __init__(self, batch_size, features):
        self.batch_size = batch_size
        self.layer_num = len(features)-1
        super(Discriminator, self).__init__()

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx+1], kernel_size=1, stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.final_layer = nn.Sequential(nn.Linear(features[-1], features[-3]),
                                         nn.Linear(features[-3], features[-5]),
                                         nn.Linear(features[-5], 1))

    def forward(self, f):
        feat = f.transpose(1,2)
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)

        out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        out = self.final_layer(out) # (B, 1)

        return out


class Generator(nn.Module):
    def __init__(self, batch_size, features, degrees, support):
        self.batch_size = batch_size
        self.layer_num = len(features)-1
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        super(Generator, self).__init__()
        
        vertex_num = 1
        self.gcn = nn.Sequential()
        for inx in range(self.layer_num):
            if inx == self.layer_num-1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=False))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=True))
            vertex_num = int(vertex_num * degrees[inx])

    def forward(self, tree):
        feat = self.gcn(tree)
        
        self.pointcloud = feat[-1]

        return self.pointcloud

    def getPointcloud(self):
        return self.pointcloud[-1]