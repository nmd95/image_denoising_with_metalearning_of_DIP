# imports
from posixpath import split
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import isfile, join
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.under_sampling import RandomUnderSampler
from keras.metrics import MeanIoU

from torch_geometric.data import Data, DataLoader
from torch_cluster import knn_graph
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import GraphNorm
from torch_cluster import fps
from collections import Counter
import torchgeometry as tgm
import pypotree
import dgl
import keras.metrics
from torch_geometric.utils import dropout_adj
import torchvision.models as models

from model_utils import knn_dgl_to_pg, random_subsample, sample_2d_with_projections, sample_2d_batch



class Pooling(torch.nn.Module):
	def __init__(self, pool_type='max'):
		self.pool_type = pool_type
		super(Pooling, self).__init__()

	def forward(self, input):
		if self.pool_type == 'max':
			return torch.max(input, 2)[0].contiguous()
		elif self.pool_type == 'avg' or self.pool_type == 'average':
			return torch.mean(input, 2).contiguous()

class AttentionalFuser(torch.nn.Module):
    def __init__(self, in_channels_2d:int, in_channels_3d:int):
        super(AttentionalFuser, self).__init__()

        torch.manual_seed(12345)

        self.in_channels_3d = in_channels_3d
        self.in_channels_2d = in_channels_2d

        self.mlp_2d = torch.nn.Linear(in_features=self.in_channels_2d, out_features=self.in_channels_3d)
        self.mlp_3d = torch.nn.Linear(in_features=self.in_channels_3d, out_features=self.in_channels_3d)
                                      
        self.mlp_attention = torch.nn.Linear(in_features=self.in_channels_3d, out_features=self.in_channels_3d)

        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()


        
    def forward(self, features_2d, features_3d):
      features_3d = features_3d.squeeze().permute(1, 0)
      downsampled_2d = self.mlp_2d(features_2d)
      mid_3d = self.mlp_3d(features_3d)
      mid_out = self.tanh(mid_3d + downsampled_2d)
      attention_map = self.sigmoid(self.mlp_attention(mid_out))
      attended_2d_features = attention_map * downsampled_2d
      output = torch.cat([features_3d, attended_2d_features], dim=1).permute(1, 0).unsqueeze(0)
      return output
      

class vgg_fuser(torch.nn.Module):
    def __init__(self):
        super(vgg_fuser, self).__init__()

        torch.manual_seed(12345)
        
        # self.params_mapping = {64: [64, 64, (4, 4), (2, 2), (1,1)], 128: [[128, 128, (4, 4), (2, 2), (1,1)], [128, 128, (4, 4), (2, 2), (1,1)], 256:[[256, 256, (4, 4), (2, 2), (1,1)], [256, 256, (4, 4), (2, 2), (1,1)], [256, 256, (4, 4), (2, 2), (1,1)]],
        #                                                                   512: [[512, 512, (4,4), (2,2), (1,1)], [512, 512, (4,6), (2,2), (1,1)], [512, 512, (4, 4), (2, 2), (1,1)], [512, 512, (4, 4), (2, 2), (1,1)], [512, 512, (4, 4), (2, 2), (1,1)]]}
        # self.layers_mapping = {64: 5, 128: 10, 256: 17, 512: 31}

        # upsamplers - part

        self.ups1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(*[64, 8, (4, 4), (2, 2), (1,1)]))
        self.ups2 = torch.nn.Sequential(torch.nn.ConvTranspose2d(*[128, 16, (4, 4), (2, 2), (1,1)]), torch.nn.ConvTranspose2d(*[16, 16, (4, 4), (2, 2), (1,1)]))
        self.ups3 = torch.nn.Sequential(torch.nn.ConvTranspose2d(*[256, 32, (4, 4), (2, 2), (1,1)]), 
                                        torch.nn.ConvTranspose2d(*[32, 32, (4, 4), (2, 2), (1,1)]), torch.nn.ConvTranspose2d(*[32, 32, (4, 4), (2, 2), (1,1)]))
        self.ups4 = torch.nn.Sequential(torch.nn.ConvTranspose2d(*[512, 64, (4, 4), (2, 2), (1,1)]), torch.nn.ConvTranspose2d(*[64, 64, (4, 6), (2, 2), (1,1)])
                                        ,torch.nn.ConvTranspose2d(*[64, 64, (4, 4), (2, 2), (1,1)]), 
                                        torch.nn.ConvTranspose2d(*[64, 64, (4, 4), (2, 2), (1,1)]), torch.nn.ConvTranspose2d(*[64, 64, (4, 4), (2, 2), (1,1)]))

        # vgg16 - part

        vgg16_pretrained = models.vgg16(pretrained=True)

        self.conv1 = vgg16_pretrained.features[0]
        self.relu = vgg16_pretrained.features[1]
        self.conv2 = vgg16_pretrained.features[2]
        self.maxpool_1 = vgg16_pretrained.features[4]
        self.conv3 = vgg16_pretrained.features[5]
        self.conv4 = vgg16_pretrained.features[7]
        self.maxpool_2 = vgg16_pretrained.features[9]
        self.conv5 = vgg16_pretrained.features[10]
        self.conv6 = vgg16_pretrained.features[12]
        self.conv7 = vgg16_pretrained.features[14]
        self.maxpool_3 = vgg16_pretrained.features[16]
        self.conv8 = vgg16_pretrained.features[17]
        self.conv9 =vgg16_pretrained.features[19]
        self.conv10 = vgg16_pretrained.features[21]
        self.maxpool_4 = vgg16_pretrained.features[23]
        self.conv11 = vgg16_pretrained.features[24]
        self.conv12 = vgg16_pretrained.features[26]
        self.conv13 = vgg16_pretrained.features[28]
        self.maxpool_5 = vgg16_pretrained.features[30]
        

        
    def forward(self, img_2d):
     
      # vgg16 - part
      out_1 = self.conv1(img_2d)
      out_1 = self.relu(out_1)
      out_1 = self.conv2(out_1)
      out_1 = self.relu(out_1)
      pooled_out_1 = self.maxpool_1(out_1)

      out_2 = self.conv3(pooled_out_1)
      out_2 = self.relu(out_2)
      out_2 = self.conv4(out_2)
      out_2 = self.relu(out_2)
      pooled_out_2 = self.maxpool_2(out_2)

      out_3 = self.conv5(pooled_out_2)
      out_3 = self.relu(out_3)
      out_3 = self.conv6(out_3)
      out_3 = self.relu(out_3)
      out_3 = self.conv7(out_3)
      out_3 = self.relu(out_3)
      pooled_out_3 = self.maxpool_3(out_3)

      out_4 = self.conv8(pooled_out_3)
      out_4 = self.relu(out_4)
      out_4 = self.conv9(out_4)
      out_4 = self.relu(out_4)
      out_4 = self.conv10(out_4)
      out_4 = self.relu(out_4)
      pooled_out_4 = self.maxpool_4(out_4)

      out_5 = self.conv11(pooled_out_4)
      out_5 = self.relu(out_5)
      out_5 = self.conv12(out_5)
      out_5 = self.relu(out_5)
      out_5 = self.conv13(out_5)
      out_5 = self.relu(out_5)
      pooled_out_5 = self.maxpool_5(out_5)

      out_fuse_1 = self.ups1(pooled_out_1)
      out_fuse_2 = self.ups2(pooled_out_2)
      out_fuse_3 = self.ups3(pooled_out_3)
      out_fuse_4 = self.ups4(pooled_out_5)

      return [out_fuse_1, out_fuse_2, out_fuse_3, out_fuse_4]

      
      
class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Message passing with "max" aggregation.
        super(PointNetLayer, self).__init__('max')
        
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels + 3, out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels)) # define the NN; out_channels= output vector that the layer in the preduce and propagate to the the same node.
        
    def forward(self, h, pos, edge_index): #edge index- tensor neighbors of node pose; h- vector features of node pose
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)
    
    def message(self, h_j, pos_j, pos_i): #updading the the graph's features by aggregating the features from j's node neighbors. func forward calls this func
        # h_j defines the features of neighboring nodes of pos_j as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_j, input], dim=-1)

        return self.mlp(input)  # Apply our final MLP.


class PointNet_Graph_Vanilla_Parallel(torch.nn.Module):
    def __init__(self):
        super(PointNet_Graph_Vanilla_Parallel, self).__init__()

        torch.manual_seed(12345) #keep the random constant after the fairst run = reproducibillity 

        self.graph_n_1 = GraphNorm(in_channels=3) #GraphNorm- normalize the graph's feature
        self.conv1 = PointNetLayer(3, 64)
        self.graph_n_2 = GraphNorm(in_channels=64)
        self.conv2 = PointNetLayer(64, 128)
        self.graph_n_3 = GraphNorm(in_channels=128)
        self.conv3 = PointNetLayer(128, 256)
        self.graph_n_4 = GraphNorm(in_channels=256)
        self.conv4 = PointNetLayer(256, 512)
        self.graph_n_5 = GraphNorm(in_channels=512)
        self.conv5 = PointNetLayer(512, 512)
        self.graph_n_6 = GraphNorm(in_channels=512)
        self.classifier = torch.nn.Linear(512, 14)

        
    def forward(self, data): #def forward(self, pos, x, batch):
        print('Inside Model:  num graphs: {}, device: {}'.format(
            data.num_graphs, data.batch.device))
        pos, x, batch = data.pos, data.x, data.batch
        # edge_index = knn_dgl_to_pg(pos, k=self.k, device=self.device).to(self.device)
        edge_index = knn_graph(pos, k=64, batch=batch, loop=True)

        h = self.graph_n_1(pos, batch)
        h = self.conv1(h=h, pos=pos, edge_index=dropout_adj(edge_index)[0]) #double code need to erase maybe 
        h = self.graph_n_2(h, batch)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h = self.graph_n_3(h, batch)
        h = h.relu()
        h = self.conv3(h=h, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h = self.graph_n_4(h, batch)
        h = h.relu()
        h = self.conv4(h=h, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h = self.graph_n_5(h, batch)
        h = h.relu()
        h = self.conv5(h=h, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h = self.graph_n_6(h, batch)
        h = h.relu()
# define each layer in the NN        

        return self.classifier(h)

class PointNet_Graph_Naive_RGB_Parallel(torch.nn.Module):
    def __init__(self):
        super(PointNet_Graph_Naive_RGB_Parallel, self).__init__()

        torch.manual_seed(12345) #keep the random constant after the fairst run = reproducibillity 

        self.conv1 = PointNetLayer(3, 64)
        self.graph_n_1 = GraphNorm(in_channels=64)
        self.conv2 = PointNetLayer(64, 128)
        self.graph_n_2 = GraphNorm(in_channels=128)
        self.conv3 = PointNetLayer(128, 256)
        self.graph_n_3 = GraphNorm(in_channels=256)
        self.conv4 = PointNetLayer(256, 512)
        self.graph_n_4 = GraphNorm(in_channels=512)
        self.conv5 = PointNetLayer(512, 512)
        self.graph_n_5 = GraphNorm(in_channels=512)
        self.classifier = torch.nn.Linear(512, 7)

        
    def forward(self, data): #def forward(self, pos, x, batch):
        # print('Inside Model:  num graphs: {}, device: {}'.format(
        #     data.num_graphs, data.batch.device))


        pos, x, batch = data.pos, data.x, data.batch
        edge_index = knn_graph(pos, k=16, batch=batch, loop=True) # was k=64

        h = self.conv1(h=x, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h = self.graph_n_1(h, batch)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h = self.graph_n_2(h, batch)
        h = h.relu()
        h = self.conv3(h=h, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h = self.graph_n_3(h, batch)
        h = h.relu()
        h = self.conv4(h=h, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h = self.graph_n_4(h, batch)
        h = h.relu()
        h = self.conv5(h=h, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h = self.graph_n_5(h, batch)
        h = h.relu()
# define each layer in the NN        

        return self.classifier(h)

        

# NoAttention == simple concat of 2d and 3d features (as opposed to passing them thru an interpolation-model)
class PointNet_Graph_MidFusion_NoAttention_Parallel(torch.nn.Module):
    def __init__(self):
        super(PointNet_Graph_MidFusion_NoAttention_Parallel, self).__init__()
        
        self.conv1 = PointNetLayer(3, 6)
        self.conv_2d_1 = torch.nn.Conv2d(3, 6, (3,3), (1, 1), (1, 1)) # these convolution parameters keep the spatial dimensions of the image
        self.graph_n_1 = GraphNorm(in_channels=12)
        self.conv2 = PointNetLayer(12, 24)
        self.conv_2d_2 = torch.nn.Conv2d(6, 12, (3,3), (1, 1), (1, 1))
        self.graph_n_2 = GraphNorm(in_channels=36)
        self.conv3 = PointNetLayer(36, 46)
        self.conv_2d_3 = torch.nn.Conv2d(12, 24, (3,3), (1, 1), (1, 1))
        self.graph_n_3 = GraphNorm(in_channels=70)
        self.conv4 = PointNetLayer(70, 128)
        self.conv_2d_4 = torch.nn.Conv2d(24, 48, (3,3), (1, 1), (1, 1))
        self.graph_n_4 = GraphNorm(in_channels=176)
        self.conv5 = PointNetLayer(176, 176)
        self.graph_n_5 = GraphNorm(in_channels=176)
        self.classifier = torch.nn.Linear(176, 7)


        
    
    def forward(self, data):
        pos, batch, imgs, projections = data.pos, data.batch, data.img_2d, data.projections
        edge_index = knn_graph(pos, k=4, batch=batch, loop=True)
        RGB = sample_2d_batch(imgs, projections, batch)
        h = self.conv1(h=RGB, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h_prime = self.conv_2d_1(imgs)
        h_prime_proj = sample_2d_batch(h_prime, projections, batch)
        h = self.graph_n_1(torch.cat([h, h_prime_proj], dim=1), batch)

        

        h = self.conv2(h=h, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h_prime = self.conv_2d_2(h_prime)
        h_prime_proj = sample_2d_batch(h_prime, projections, batch)
        h = self.graph_n_2(torch.cat([h, h_prime_proj], dim=1))

        h = self.conv3(h=h, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h_prime = self.conv_2d_3(h_prime)
        h_prime_proj = sample_2d_batch(h_prime, projections, batch)
        h = self.graph_n_3(torch.cat([h, h_prime_proj], dim=1), batch)

        h = self.conv4(h=h, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h_prime = self.conv_2d_4(h_prime)
        h_prime_proj = sample_2d_batch(h_prime, projections, batch)
        h = self.graph_n_4(torch.cat([h, h_prime_proj], dim=1), batch)

        h = self.conv5(h=h, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h = self.graph_n_5(h, batch)


        return self.classifier(h)

class demo_points_img_parallel_batcher(torch.nn.Module):
    def __init__(self):
        super(demo_points_img_parallel_batcher, self).__init__()
        
        self.conv1 = PointNetLayer(3, 6)
        self.conv_2d_1 = torch.nn.Conv2d(3, 6, (3,3), (1, 1), (1, 1))
        self.graph_n_1 = GraphNorm(in_channels=12)
        self.conv2 = PointNetLayer(12, 24)
        self.conv_2d_2 = torch.nn.Conv2d(6, 12, (3,3), (1, 1), (1, 1))
        self.graph_n_2 = GraphNorm(in_channels=36)
        self.conv3 = PointNetLayer(36, 46)
        self.conv_2d_3 = torch.nn.Conv2d(12, 24, (3,3), (1, 1), (1, 1))
        self.graph_n_3 = GraphNorm(in_channels=70)
        self.conv4 = PointNetLayer(70, 128)
        self.conv_2d_4 = torch.nn.Conv2d(24, 48, (3,3), (1, 1), (1, 1))
        self.graph_n_4 = GraphNorm(in_channels=176)
        self.conv5 = PointNetLayer(176, 176)
        self.graph_n_5 = GraphNorm(in_channels=176)
        self.classifier = torch.nn.Linear(176, 14)


        
    
    def forward(self, data):
        pos, batch, imgs, projections = data.pos, data.batch, data.img_2d, data.projections
        edge_index = knn_graph(pos, k=4, batch=batch, loop=True)
        RGB = sample_2d_batch(imgs, projections, batch)
        h = self.conv1(h=RGB, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h_prime = self.conv_2d_1(imgs)
        h_prime_proj = sample_2d_batch(h_prime, projections, batch)
        h = self.graph_n_1(torch.cat([h, h_prime_proj], dim=1), batch)

        

        h = self.conv2(h=h, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h_prime = self.conv_2d_2(h_prime)
        h_prime_proj = sample_2d_batch(h_prime, projections, batch)
        h = self.graph_n_2(torch.cat([h, h_prime_proj], dim=1))

        h = self.conv3(h=h, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h_prime = self.conv_2d_3(h_prime)
        h_prime_proj = sample_2d_batch(h_prime, projections, batch)
        h = self.graph_n_3(torch.cat([h, h_prime_proj], dim=1), batch)

        h = self.conv4(h=h, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h_prime = self.conv_2d_4(h_prime)
        h_prime_proj = sample_2d_batch(h_prime, projections, batch)
        h = self.graph_n_4(torch.cat([h, h_prime_proj], dim=1), batch)

        h = self.conv5(h=h, pos=pos, edge_index=dropout_adj(edge_index)[0])
        h = self.graph_n_5(h, batch)


        return self.classifier(h)




        # print(batch)
        # print(batch.shape)
        # print(projections.shape)
        # print(imgs.shape)
        # print(data.num_graphs)
        # print(data.num_nodes)
        # sampled_imgs = sample_2d_batch(imgs, projections, batch)
        # print(sampled_imgs.shape)





class PointNet_MidFusion(torch.nn.Module):

  def __init__(self, fuser, emb_dims=256):
	  # emb_dims:			Embedding Dimensions for PointNet.
	  # input_shape:		Shape of Input Point Cloud (b: batch, c: channels, n: no of points)
    super(PointNet_MidFusion, self).__init__()
   
    self.fuser = fuser
    self.emb_dims = emb_dims
    self.pooling = Pooling('max')
    self.conv1 = torch.nn.Conv1d(3 + 3, 8, 1) # +3 for RGB!
    self.att_fuser1 = AttentionalFuser(in_channels_2d=8, in_channels_3d=8)
    self.bn1 = torch.nn.BatchNorm1d(16)
    self.conv2 = torch.nn.Conv1d(16, 16, 1)
    self.att_fuser2 = AttentionalFuser(in_channels_2d=16, in_channels_3d=16)
    self.bn2 = torch.nn.BatchNorm1d(32)
    self.conv3 = torch.nn.Conv1d(32, 32, 1)
    self.att_fuser3 = AttentionalFuser(in_channels_2d=32, in_channels_3d=32)
    self.bn3 = torch.nn.BatchNorm1d(64)
    self.conv4 = torch.nn.Conv1d(64, 64, 1)
    self.att_fuser4 = AttentionalFuser(in_channels_2d=64, in_channels_3d=64)
    self.bn4 = torch.nn.BatchNorm1d(128)
    self.conv5 = torch.nn.Conv1d(128, self.emb_dims, 1)
    self.bn5 = torch.nn.BatchNorm1d(self.emb_dims)
    self.relu = torch.nn.ReLU()

		# self.att_fuser1 = AttentionalFuser(in_channels_2d=64, in_channels_3d=32)
    # self.att_fuser2 = AttentionalFuser(in_channels_2d=64, in_channels_3d=32)

  def forward(self, sample): # batch_size*(num_points*(x, y, z, class, pixLocs(x), pixLocs(y), time, R, G, B), image_2d)
		# input_data: 		Point Cloud having shape input_shape: (batch * channels * num_in_points)
		# output:			PointNet features (Batch x emb_dims)

    sample = (sample[0].squeeze(), sample[1].squeeze())
    pos = sample[0][:, 0:3]
    y = sample[0][:, 3:4]
    projections = sample[0][:, 4:6]
    num_points = sample[0].shape[0]
    RGB = sample[0][:, 7:10]
    input_data = pos
    img_2d = sample[1]
    init_features = torch.cat([pos, RGB], dim=1).permute(1, 0).unsqueeze(0) # of shape (batch_size, channels, #points)

    feature_maps_2d = self.fuser(img_2d.unsqueeze(0))
    feature_maps_samples = [sample_2d_with_projections(fmap, projections=projections).squeeze() for fmap in feature_maps_2d]
    
    output = init_features
    output = self.conv1(output)
    output = self.relu(output)
    output = self.att_fuser1(features_2d=feature_maps_samples[0], features_3d=output)
    output = self.bn1(output)
    
    output = self.conv2(output)
    output = self.relu(output)
    output = self.att_fuser2(features_2d=feature_maps_samples[1], features_3d=output)
    output = self.bn2(output)
    

    output = self.conv3(output)
    output = self.relu(output)
    output = self.att_fuser3(features_2d=feature_maps_samples[2], features_3d=output)
    output = self.bn3(output)
    
    point_feature = output

    output = self.conv4(output)             
    output = self.relu(output)
    output = self.att_fuser4(features_2d=feature_maps_samples[3], features_3d=output)
    output = self.bn4(output)
    
    output = self.conv5(output)
    output = self.relu(output)
    output = self.bn5(output)
    
    
    output = self.pooling(output)
    output = output.view(-1, self.emb_dims, 1).repeat(1, 1, num_points)
    return torch.cat([output, point_feature], 1) #output is of shape (batch_size, features(concat of point_f+global_descriptor, #points))


