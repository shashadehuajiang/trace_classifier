# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:03:50 2020

@author: sahua
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from mylib import name_generator
import random
import time
import pytorch_warmup 
import copy
import math


class Net(nn.Module):
    def __init__(self,cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.web_num = cfg.CLASS_NUM
        self.emb_num = cfg.EMB_NUM
        self.f_emb_num = cfg.F_EMB_NUM
        self.feature_size = cfg.FEATURE_SIZE
        self.usebn = cfg.USEBN_FALG
        self.usedo = cfg.DROPOUT_FLAG
        self.device = cfg.DEVICE
        self.train_steps = 0

        self.gen_layers()


    def gen_layers(self):
        if self.cfg.PACTETCNN:
            #CNN net
            self.cn_conv1 = nn.Conv1d(in_channels=self.cfg.PACKTE_DIM, out_channels=10, kernel_size=5)
            self.cn_conv2 = nn.Conv1d(in_channels=10, out_channels=int(self.feature_size/2), kernel_size=5)
            self.cn_conv3 = nn.Conv1d(in_channels=int(self.feature_size/2), out_channels=int(self.feature_size/1), kernel_size=5)
            #self.cn_conv4 = nn.Conv1d(in_channels=int(feature_size/2), out_channels=feature_size, kernel_size=5)
            self.cn_pool1 = nn.MaxPool1d(3, stride=3,padding=1)
            self.cn_pool2 = nn.MaxPool1d(3, stride=3,padding=1)
            self.cn_pool3 = nn.MaxPool1d(3, stride=3,padding=1)
            #self.cn_pool4 = nn.MaxPool1d(3, stride=2,padding=1)
        else:
            self.cn_conv = nn.Conv1d(in_channels=self.cfg.PACKTE_DIM, out_channels=int(self.feature_size/1), kernel_size=1)
            if self.cfg.PACKET2FLOW == '2dCNN':
                self.cn_conv = nn.Conv2d(in_channels=self.cfg.PACKTE_DIM, out_channels=int(self.feature_size/1), kernel_size=(1,1))

        if (self.cfg.PACKET2FLOW == 'ATT' or self.cfg.PACKET2FLOW == 'LSTM+ATT') and self.cfg.TRANSFORMER:
            self.p_trans_en = nn.TransformerEncoderLayer(d_model=self.feature_size, nhead=4)
            self.p_trans = nn.TransformerEncoder(self.p_trans_en, num_layers=2)
            self.f_trans_en = nn.TransformerEncoderLayer(d_model=self.feature_size, nhead=4)
            self.f_trans = nn.TransformerEncoder(self.f_trans_en, num_layers=2)


        #generate session feature attention net
        if self.cfg.PACKET2FLOW == 'ATT':
            self.emb1 = nn.Embedding(self.emb_num, self.feature_size) # pactet2flow
            self.f_emb1 = nn.Embedding(self.f_emb_num, self.feature_size)
            self.f_wk_linear1 = nn.Linear(self.feature_size, self.feature_size)
            self.f_wv_linear1 = nn.Linear(self.feature_size, self.feature_size)
            #self.f_linear3 = nn.Linear(int(feature_size*1.5), feature_size)
            self.f_conv1 = nn.Conv1d(in_channels=self.f_emb_num, out_channels=10, kernel_size=1)
            self.f_conv2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1)
        elif self.cfg.PACKET2FLOW == 'LSTM':
            self.f_lstm = nn.LSTM(self.feature_size, self.feature_size, 3, bidirectional=True)
        elif self.cfg.PACKET2FLOW == 'TREE':
            self.f_linear1 = nn.Linear(self.feature_size*2, int(self.feature_size*1.75))
            self.f_linear2 = nn.Linear(int(self.feature_size*1.75), int(self.feature_size*1.5))
            self.f_linear3 = nn.Linear(int(self.feature_size*1.5), self.feature_size)
        elif self.cfg.PACKET2FLOW == 'LSTM+ATT':
            self.emb1 = nn.Embedding(self.emb_num, self.feature_size) # pactet2flow
            self.f_lstm = nn.LSTM(self.feature_size, self.feature_size, 3, bidirectional=True)
            self.f_emb1 = nn.Embedding(self.f_emb_num, self.feature_size)
            self.f_wk_linear1 = nn.Linear(self.feature_size, self.feature_size)
            self.f_wv_linear1 = nn.Linear(self.feature_size, self.feature_size)
            #self.f_linear3 = nn.Linear(int(feature_size*1.5), feature_size)
            self.f_conv1 = nn.Conv1d(in_channels=self.f_emb_num, out_channels=10, kernel_size=1)
            self.f_conv2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1)
        elif self.cfg.PACKET2FLOW == '1dCNN':
            flow_length = self.cfg.FIX_M1_CNN_LENGTH
            self.f_cnn_layers = []
            cnn_layer_num = int(math.log(flow_length,3))
            print('cnn_layer_num:',cnn_layer_num)

            for _ in range(cnn_layer_num+1):
                cnn_layer = nn.Conv1d(in_channels=self.f_emb_num, out_channels=self.f_emb_num, kernel_size=3,padding=1)
                pooling_layer = nn.MaxPool1d(3, stride=3,padding=1)
                self.f_cnn_layers.append(cnn_layer.to(self.device))
                self.f_cnn_layers.append(nn.LeakyReLU().to(self.device))
                self.f_cnn_layers.append(pooling_layer.to(self.device))

        elif self.cfg.PACKET2FLOW == '2dCNN':
            x_size = self.cfg.SIZE_2dCNN

            self.f_cnn_layers = []
            cnn_layer_num = int(math.log(x_size,3))
            print('cnn_layer_num:',cnn_layer_num)

            for _ in range(cnn_layer_num+1):
                cnn_layer = nn.Conv2d(in_channels=self.f_emb_num, out_channels=self.f_emb_num, kernel_size=(3,3),padding=(1,1))
                pooling_layer = nn.MaxPool2d((3,3), stride=(3,3),padding=(1,1))
                self.f_cnn_layers.append(cnn_layer.to(self.device))
                self.f_cnn_layers.append(nn.LeakyReLU().to(self.device))
                self.f_cnn_layers.append(pooling_layer.to(self.device))

        else:
            os.exit()
        
        
        if self.cfg.LNORM_FLAG:
            self.lnorm1 = nn.LayerNorm(self.feature_size)


        if self.cfg.FLOW2TRACE == 'ATT':
            self.emb2 = nn.Embedding(self.emb_num, self.feature_size) # flow2trace
            #attention net
            self.ss_wk_linear1 = nn.Linear(self.feature_size, self.feature_size)
            self.ss_wv_linear1 = nn.Linear(self.feature_size, self.feature_size)
        elif self.cfg.FLOW2TRACE == 'LSTM':
            self.t_lstm = nn.LSTM(self.feature_size, self.feature_size, 1, bidirectional=True)
        elif self.cfg.FLOW2TRACE == 'TREE':
            self.t_linear1 = nn.Linear(self.feature_size*2, int(self.feature_size*1.75))
            self.t_linear2 = nn.Linear(int(self.feature_size*1.75), int(self.feature_size*1.5))
            self.t_linear3 = nn.Linear(int(self.feature_size*1.5), self.feature_size)
        elif self.cfg.FLOW2TRACE == 'LSTM+ATT':
            self.emb2 = nn.Embedding(self.emb_num, self.feature_size)
            #attention net
            self.ss_wk_linear1 = nn.Linear(self.feature_size, self.feature_size)
            self.ss_wv_linear1 = nn.Linear(self.feature_size, self.feature_size)
            self.t_lstm = nn.LSTM(self.feature_size, self.feature_size, 1, bidirectional=True)
        else:
            os.exit()

        #classifer net
        self.c_conv1 = nn.Conv1d(in_channels=self.emb_num, out_channels=self.emb_num, kernel_size=1)
        self.c_conv2 = nn.Conv1d(in_channels=self.emb_num, out_channels=1, kernel_size=1)
        self.c_linear1 = nn.Linear(self.feature_size, 100)
        self.c_linear2 = nn.Linear(100, 100)
        self.c_linear3 = nn.Linear(100, 50)
        self.c_linear4 = nn.Linear(50, self.web_num)
        
        if self.usebn:
            self.c_bn1 = nn.BatchNorm1d(self.emb_num,momentum = 0.01)
            self.c_bn2 = nn.BatchNorm1d(self.emb_num,momentum = 0.01)
            self.c_bn3 = nn.BatchNorm1d(100,momentum = 0.01)
            self.c_bn4 = nn.BatchNorm1d(100,momentum = 0.01)

        #mtrain
        if self.cfg.USEMM_FLAG:
            self.mt_linear1 = nn.Linear(self.feature_size, self.web_num)
            self.mt2_linear1 = nn.Linear(self.feature_size, self.web_num)
        
    def flow_feature_generator(self,tensor_vector,device):
        if self.cfg.PACKET2FLOW == 'ATT':
            q = torch.arange(0,self.f_emb_num).view(-1).long()
            q = q.to(device)
            q = self.f_emb1(q)
            #print('q.size()',q.size())
            #print(tensor_vector.size())
            
            x = tensor_vector[0]
            x = x.t()
            key = self.f_wk_linear1(x).t()
            value = self.f_wv_linear1(x)
            
            #softmax(q*kT/sqrt(size)) * v
            attention_a = q.mm(key) / (self.feature_size**0.5)
            attention_soft = F.softmax(attention_a,dim=1).to(device)
            attention_x = F.leaky_relu(attention_soft.mm(value))
            attention_x = attention_x.view(-1,self.f_emb_num,self.feature_size)
            
            x = attention_x
            if self.cfg.TRANSFORMER:
                x = x.permute(1,0,2)
                x = self.p_trans(x)
                x = x.permute(1,0,2)

            x = F.leaky_relu(self.f_conv1(x))
            x = F.leaky_relu(self.f_conv2(x))
            
            return x.view(1,self.feature_size)

        elif self.cfg.PACKET2FLOW == 'LSTM':
            x = tensor_vector.permute(2,0,1)
            out, (h_n, c_n) = self.f_lstm(x)
            x = h_n[-1, :, :] +  h_n[-2, :, :]

            return x.view(1,self.feature_size)

        elif self.cfg.PACKET2FLOW == 'TREE':
            if tensor_vector.size(2) == 1:
                va = tensor_vector.view(-1,self.feature_size)
                return va
            
            middle_index = int(tensor_vector.size(2)/2)
            x_list1 = tensor_vector[:,:,:middle_index]
            x_list2 = tensor_vector[:,:,middle_index:]
            
            x1 = self.flow_feature_generator(x_list1,device)
            x2 = self.flow_feature_generator(x_list2,device)
            
            x = torch.cat((x1,x2),1)
            #print('x1',x1.size())
            #print(x2.size())
            #os.exit()
            x = F.leaky_relu(self.f_linear1(x))
            x = F.leaky_relu(self.f_linear2(x))
            x = F.leaky_relu(self.f_linear3(x))

            return x

        elif self.cfg.PACKET2FLOW == 'LSTM+ATT':
            x = tensor_vector.permute(2,0,1)
            out, (h_n, c_n) = self.f_lstm(x)
            x = out.permute(1,0,2)
            x = x[:,:,:int(x.shape[2]/2)]  +  x[:,:,int(x.shape[2]/2):]
            
            q = torch.arange(0,self.f_emb_num).view(-1).long()
            q = q.to(device)
            q = self.f_emb1(q)
            #print('q.size()',q.size())
            #print(tensor_vector.size())
            
            x = x[0]
            #x = x.t()
            key = self.f_wk_linear1(x).t()
            value = self.f_wv_linear1(x)
            
            #softmax(q*kT/sqrt(size)) * v
            attention_a = q.mm(key) / (self.feature_size**0.5)
            attention_soft = F.softmax(attention_a,dim=1).to(device)
            attention_x = F.leaky_relu(attention_soft.mm(value))
            attention_x = attention_x.view(-1,self.f_emb_num,self.feature_size)
            
            x = attention_x
            if self.cfg.TRANSFORMER:
                x = x.permute(1,0,2)
                x = self.p_trans(x)
                x = x.permute(1,0,2)

            x = F.leaky_relu(self.f_conv1(x))
            x = F.leaky_relu(self.f_conv2(x))
            
            return x.view(1,self.feature_size)

        elif self.cfg.PACKET2FLOW == '1dCNN':
            x = tensor_vector

            for layer in self.f_cnn_layers:
                x = layer(x)


            return x.view(1,self.feature_size)
        
        elif self.cfg.PACKET2FLOW == '2dCNN':
            x = tensor_vector
            for layer in self.f_cnn_layers:
                x = layer(x)
                
            return x.view(1,self.feature_size)


            
    
    def seq_2_vector(self,seq_in,device):
        #print('seq_in.shape',seq_in.shape)
        if self.cfg.PACKET2FLOW == '2dCNN':
            tensor_in = torch.from_numpy(np.array(seq_in,dtype = np.float32))
            variable_x = Variable(tensor_in,requires_grad=False).to(device)
            variable_x = variable_x.view(-1,self.cfg.PACKTE_DIM,self.cfg.SIZE_2dCNN,self.cfg.SIZE_2dCNN)
        else:
            tensor_in = torch.from_numpy(np.array(seq_in,dtype = np.float32).swapaxes(0, 1))
            variable_x = Variable(tensor_in,requires_grad=False).to(device)
            variable_x = variable_x.view(-1,self.cfg.PACKTE_DIM,len(seq_in))
        
        pad_to_size = 5 + 2
        
        if self.cfg.PACTETCNN:
            if variable_x.size(2)<pad_to_size:
                pad = nn.ConstantPad1d((0,pad_to_size-variable_x.size(2)),value = 0)
                variable_x = pad(variable_x)
            
            #print(variable_x.size())
            #Conv net
            variable_x = self.cn_pool1(F.leaky_relu(self.cn_conv1(variable_x)))
            
            if variable_x.size(2)<pad_to_size:
                pad = nn.ConstantPad1d((0,pad_to_size-variable_x.size(2)),value = 0)
                variable_x = pad(variable_x)
            
            #print(variable_x.size())
            #Conv net
            variable_x = self.cn_pool2(F.leaky_relu(self.cn_conv2(variable_x)))
            
            if variable_x.size(2)<pad_to_size:
                pad = nn.ConstantPad1d((0,pad_to_size-variable_x.size(2)),value = 0)
                variable_x = pad(variable_x)
            variable_x = self.cn_pool3(F.leaky_relu(self.cn_conv3(variable_x)))
        
        else:
            variable_x = self.cn_conv(variable_x)

        if self.cfg.USEMM_FLAG:
            self.mt_cache = []
            mt_in = self.mt2_linear1(variable_x.view(-1,self.feature_size))
            for i in range(mt_in.size(0)):
                self.mt_cache.append(mt_in[i:i+1])
        
        return self.flow_feature_generator(variable_x,device)
    
    
    def trace_feature_generator(self,x):
        if self.cfg.FLOW2TRACE == 'ATT':
            #特征向量化
            query = torch.arange(0,self.emb_num).view(-1).long()
            query = query.to(self.device)
            query = self.emb2(query)

            key = self.ss_wk_linear1(x).t()
            value = self.ss_wv_linear1(x)
            
            #softmax(q*kT/sqrt(size)) * v
            attention_a = query.mm(key) / (self.feature_size**0.5)
            attention_soft = F.softmax(attention_a,dim=1).to(self.device)
            attention_x = F.leaky_relu(attention_soft.mm(value))
            attention_x = attention_x.view(-1,self.emb_num,self.feature_size)

            x = attention_x
            if self.cfg.TRANSFORMER:
                x = x.permute(1,0,2)
                x = self.f_trans(x)
                x = x.permute(1,0,2)

            final_x = x.view(1,self.emb_num,-1)

            return final_x

        elif self.cfg.FLOW2TRACE == 'LSTM':
            x = x.view(x.shape[0],1,self.feature_size)
            x, (h_n, c_n) = self.t_lstm(x)
            x = h_n[-1, :, :] +  h_n[-2, :, :]

            return x

        elif self.cfg.FLOW2TRACE == 'TREE':
            if x.shape[0] == 1:
                return x
            
            middle_index = int(x.shape[0]/2)
            x_list1 = x[:middle_index,:]
            x_list2 = x[middle_index:,:]
            
            x1 = self.trace_feature_generator(x_list1)
            x2 = self.trace_feature_generator(x_list2)
            
            x = torch.cat((x1,x2),1)
            
            x = F.leaky_relu(self.t_linear1(x))
            x = F.leaky_relu(self.t_linear2(x))
            x = F.leaky_relu(self.t_linear3(x))

            return x

        elif self.cfg.FLOW2TRACE == 'LSTM+ATT':
            # lstm
            if x.shape[0]>0:
                x = x.view(x.shape[0],1,self.feature_size)
                x, (h_n, c_n) = self.t_lstm(x)
                x = x.view(x.shape[0],self.feature_size,-1)
                x = x[:,:,0]  +  x[:,:,1]
            
            #特征向量化
            query = torch.arange(0,self.emb_num).view(-1).long()
            query = query.to(self.device)
            query = self.emb2(query)

            key = self.ss_wk_linear1(x).t()
            value = self.ss_wv_linear1(x)
            
            #softmax(q*kT/sqrt(size)) * v
            attention_a = query.mm(key) / (self.feature_size**0.5)
            attention_soft = F.softmax(attention_a,dim=1).to(self.device)
            attention_x = F.leaky_relu(attention_soft.mm(value))
            attention_x = attention_x.view(-1,self.emb_num,self.feature_size)

            x = attention_x
            if self.cfg.TRANSFORMER:
                x = x.permute(1,0,2)
                x = self.f_trans(x)
                x = x.permute(1,0,2)

            final_x = x.view(1,self.emb_num,-1)

            return final_x


    def forward(self, x_in,device):
        x = torch.zeros((len(x_in),self.feature_size)).to(device)
        for i_x in range(len(x_in)):
            flow_feature = self.seq_2_vector(x_in[i_x],device)
            x[i_x] = flow_feature
        
        if self.cfg.LNORM_FLAG:
            x = x.view(-1,len(x_in),self.feature_size)
            x = self.lnorm1(x)
            x = x.view(len(x_in),self.feature_size)


        final_x = self.trace_feature_generator(x)

        final_x = self.forward2(final_x)
        
        return final_x
    


    def get_last_layer(self, x_in):
        x = torch.zeros((len(x_in),self.feature_size)).to(self.device)
        for i_x in range(len(x_in)):
            flow_feature = self.seq_2_vector(x_in[i_x],self.device)
            x[i_x] = flow_feature
        
        if self.cfg.LNORM_FLAG:
            x = x.view(-1,len(x_in),self.feature_size)
            x = self.lnorm1(x)
            x = x.view(len(x_in),self.feature_size)

        final_x = self.trace_feature_generator(x)
        x_in = final_x

        if self.usebn:
            if self.cfg.FLOW2TRACE == 'ATT' or self.cfg.FLOW2TRACE == 'LSTM+ATT':
                final_x = F.leaky_relu(self.c_conv1(self.c_bn1(x_in)))
                final_x = F.leaky_relu(self.c_conv2(self.c_bn2(final_x)))
                final_x = final_x.view(-1,self.feature_size)
                #print('final_x.shape',final_x.shape)
            else:
                final_x = x_in

            if self.usedo:
                final_x = F.leaky_relu(self.c_bn3(F.dropout(self.c_linear1(final_x))))
                final_x = F.leaky_relu(self.c_bn4(F.dropout(self.c_linear2(final_x))))
                final_x = F.leaky_relu(F.dropout(self.c_linear3(final_x)))
            else:
                final_x = F.leaky_relu(self.c_bn3(self.c_linear1(final_x)))
                final_x = F.leaky_relu(self.c_bn4(self.c_linear2(final_x)))
                final_x = F.leaky_relu(self.c_linear3(final_x))
        else:
            final_x = F.leaky_relu(self.c_conv1(x_in))
            final_x = F.leaky_relu(self.c_conv2(final_x))
            final_x = final_x.view(-1,self.feature_size)
            
            if self.usedo:
                final_x = F.leaky_relu(F.dropout(self.c_linear1(final_x)))
                final_x = F.leaky_relu(F.dropout(self.c_linear2(final_x)))
                final_x = F.leaky_relu(F.dropout(self.c_linear3(final_x)))
            else:
                final_x = F.leaky_relu(self.c_linear1(final_x))
                final_x = F.leaky_relu(self.c_linear2(final_x))
                final_x = F.leaky_relu(self.c_linear3(final_x))
        
        return final_x



    def forward1(self, x_in,device,require_mt = False):
        flow_mt_result1 = []
        flow_mt_result2 = []
        
        x = torch.zeros((len(x_in),self.feature_size)).to(device)
        for i_x in range(len(x_in)):
            flow_feature = self.seq_2_vector(x_in[i_x],device)
            if require_mt:
                flow_mt_result1.append(self.mt_linear1(flow_feature))
                flow_mt_result2.extend(self.mt_cache)
            x[i_x] = flow_feature
        
        if self.cfg.LNORM_FLAG:
            x = x.view(-1,len(x_in),self.feature_size)
            x = self.lnorm1(x)
            x = x.view(len(x_in),self.feature_size)

        final_x = self.trace_feature_generator(x)
        
        if require_mt:
            return (flow_mt_result1,flow_mt_result2,final_x)
        else:
            return final_x
        
    
    def forward2(self,x_in):
        if self.usebn:
            if self.cfg.FLOW2TRACE == 'ATT' or self.cfg.FLOW2TRACE == 'LSTM+ATT':
                final_x = F.leaky_relu(self.c_conv1(self.c_bn1(x_in)))
                final_x = F.leaky_relu(self.c_conv2(self.c_bn2(final_x)))
                final_x = final_x.view(-1,self.feature_size)
                #print('final_x.shape',final_x.shape)
            else:
                final_x = x_in

            if self.usedo:
                final_x = F.leaky_relu(self.c_bn3(F.dropout(self.c_linear1(final_x))))
                final_x = F.leaky_relu(self.c_bn4(F.dropout(self.c_linear2(final_x))))
                final_x = F.leaky_relu(F.dropout(self.c_linear3(final_x)))
                final_x = self.c_linear4(final_x)
            else:
                final_x = F.leaky_relu(self.c_bn3(self.c_linear1(final_x)))
                final_x = F.leaky_relu(self.c_bn4(self.c_linear2(final_x)))
                final_x = F.leaky_relu(self.c_linear3(final_x))
                final_x = self.c_linear4(final_x)
        else:
            final_x = F.leaky_relu(self.c_conv1(x_in))
            final_x = F.leaky_relu(self.c_conv2(final_x))
            final_x = final_x.view(-1,self.feature_size)
            
            if self.usedo:
                final_x = F.leaky_relu(F.dropout(self.c_linear1(final_x)))
                final_x = F.leaky_relu(F.dropout(self.c_linear2(final_x)))
                final_x = F.leaky_relu(F.dropout(self.c_linear3(final_x)))
                final_x = self.c_linear4(final_x)
            else:
                final_x = F.leaky_relu(self.c_linear1(final_x))
                final_x = F.leaky_relu(self.c_linear2(final_x))
                final_x = F.leaky_relu(self.c_linear3(final_x))
                final_x = self.c_linear4(final_x)

        return final_x


    def data_enhancement(self,X):
        if self.cfg.PACKET2FLOW == '2dCNN':
            return X # not ready yet

        if self.cfg.PACKET2FLOW == '1dCNN' and self.cfg.SCALE_1dCNN:
            return X # not ready yet

        X_new = copy.deepcopy(X)
        
        for i_f in range(len(X_new)):
            for i_v in range(len(X_new[i_f])):
                X_new[i_f][i_v] = X_new[i_f][i_v][0:2]

        rand_num = random.random()

        if rand_num>0.5 and rand_num<0.7:
            # 随机丢弃
            for i_f in range(len(X_new)):
                for i_v in range(len(X_new[i_f])-1,-1,-1):
                    rand_threshold = random.random()*0.1
                    if random.random() < rand_threshold:
                        del X_new[i_f][i_v]

        if rand_num>0.7 and rand_num<0.8:
            # 时间平移（保序）
            for i_f in range(len(X_new)):
                for i_v in range(1,len(X_new[i_f])):
                    if i_v == len(X_new[i_f]) - 1:
                        max_size = X_new[i_f][i_v][0]/2
                        random_biaos = (random.random()*2-1)*max_size
                        X_new[i_f][i_v][0] -= random_biaos
                    else:
                        max_size = min(X_new[i_f][i_v][0],X_new[i_f][i_v+1][0])
                        random_biaos = (random.random()*2-1)*max_size
                        X_new[i_f][i_v][0] -= random_biaos
                        X_new[i_f][i_v+1][0] += random_biaos
        
        if rand_num>0.8 and rand_num<1:
            # 前后随机cut
            max_time_list = [max([X_new[i_f][i_v][0] for i_v in range(0,len(X_new[i_f]))]) for i_f in range(len(X_new))] 
            max_time = max(max_time_list)
            
            startcut = (random.random()*0.2) * max_time
            endcut = (1 - random.random()*0.2) * max_time
            for i_f in range(len(X_new)-1,-1,-1):
                for i_v in range(len(X_new[i_f])-1,-1,-1):
                    if X_new[i_f][i_v][0] < startcut or X_new[i_f][i_v][0] > endcut:
                        del X_new[i_f][i_v]
                if len(X_new[i_f]) == 0:
                    del X_new[i_f]



        for i_f in range(len(X_new)):
            for i_v in range(len(X_new[i_f])):
                # 添加绝对时间
                if i_v>=1:
                    time_sum = X_new[i_f][i_v][0] + X_new[i_f][i_v-1][-1]
                    X_new[i_f][i_v] = np.r_[X_new[i_f][i_v],time_sum]
                else:
                    X_new[i_f][i_v] = np.r_[X_new[i_f][i_v],0]
                # 添加顺序，并clip
                id_pad = min(i_v/100,1)
                X_new[i_f][i_v] = np.r_[X_new[i_f][i_v],id_pad]
                # 添加百分比
                percentage = i_v/len(X_new[i_f])
                X_new[i_f][i_v] = np.r_[X_new[i_f][i_v],percentage]

        return X

    
    def rand_seq_keep_list_add(self,list1,list2):
        return_list = []
        random_points1 = [random.random() for i in range(len(list1))]
        random_points2 = [random.random() for i in range(len(list2))]
        
        random_points1.sort()
        random_points2.sort()
        i = 0 
        j = 0
        while(True):
            #print(i,j)
            if i >= len(list1) and j >= len(list2):
                break
            
            if i == len(list1):
                return_list.append(list2[j])
                j += 1
                continue
            if j == len(list2):
                return_list.append(list1[i])
                i += 1
                continue
            
            if random_points1[i] < random_points2[j]:
                return_list.append(list1[i])
                i += 1
            else:
                return_list.append(list2[j])
                j += 1
                
        return return_list  

    def my_train(self,data_in,optimizer,batch,usemm,kth = 0):
        '''
        Parameters
        ----------
        data_in : list
            数据的输入
        batch : int
            每次训练个数
        model : nn.Module
            模型
        device : string
            cpu还是gpu
        batch : 批训练
            批训练个数，一次训练多少数据

        Returns
        -------
        一个训练完的分类器
        '''
        #设定参数可变
        self.train()
        
        # 打乱数据使用的下标
        random_index = random.sample(range(len(data_in)), len(data_in))
        
        label_dict = {} # 索引表，方便时候随机使用
        for i in range(len(data_in)):
            if data_in[i][1] not in label_dict.keys():
                label_dict[data_in[i][1]] = []
                label_dict[data_in[i][1]].append(i)
            else:
                label_dict[data_in[i][1]].append(i)
    
        if self.cfg.WF_MIX_PAGE and self.cfg.DATASET_NAME == 'wf':
            bg_key = -1
            bg_size = -1
            for key in label_dict.keys():
                if len(label_dict[key]) > bg_size:
                    bg_key = key
                    bg_size = len(label_dict[key]) 
            print('bg_key',bg_key)


        if len(label_dict.keys())<2:
            print('种类不足两类，不需要进行分类....')
            return
        
        sum_loss = 1
        
        batch_x_cache = None
        batch_y_cache = None
        batch_mt1_cache = []
        batch_mt2_cache = []
        batch_mty1_cache = []
        batch_mty2_cache = []
        
        y_true = []
        y_pred = []

        time0 = time.time()
        for i in range(len(data_in)):
            
            try:

                # 释放变量显存
                torch.cuda.empty_cache()

                rand_i = random.choice(label_dict[random.choice(list(label_dict.keys()))])
                X = data_in[rand_i][0]
                Y = data_in[rand_i][1]
                #X = data_in[random_index[i]][0]    # 一个训练数据
                #Y = data_in[random_index[i]][1]    # 一个训练标签
                y_true.append(Y)
                
                if self.cfg.DATA_ENHANCEMENT_FLAG:
                    if random.random()<0.5:
                        X = self.data_enhancement(X)

                # 复制X
                input_X = copy.deepcopy(X)

                # 中途训练数据
                Label = torch.tensor([Y]).to(self.device)
                if self.cfg.WF_MIX_PAGE and self.cfg.DATASET_NAME == 'wf':
                    # 加入背景流量(随机0-2)
                    for _ in range(random.randint(0,2)):
                        rand_bg_i = random.randint(0,bg_size-1)
                        input_X = self.rand_seq_keep_list_add(input_X,data_in[label_dict[bg_key][rand_bg_i]][0])

                if usemm:
                    (flow_mt_result1,flow_mt_result2,middle_result) = self.forward1(input_X,self.device,usemm)
                else:
                    middle_result = self.forward1(input_X,self.device,usemm)
                    
                if batch_x_cache == None:
                    batch_x_cache = middle_result
                    batch_y_cache = Label
                else:
                    batch_x_cache = torch.cat((batch_x_cache,middle_result),0)
                    batch_y_cache = torch.cat((batch_y_cache,Label))
                
                if usemm:
                    batch_mt1_cache.extend(flow_mt_result1)
                    batch_mt2_cache.extend(flow_mt_result2)
                    batch_mty1_cache.extend(list(Label for i in flow_mt_result1))
                    batch_mty2_cache.extend(list(Label for i in flow_mt_result2))
                
                
                if (i+1) % batch == 0:  # 按照batch进行处理
                    predict = self.forward2(batch_x_cache)
                    pred = predict.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    loss = F.cross_entropy(predict,batch_y_cache)
                    
                    if usemm:
                        loss.backward(retain_graph=True)
                        loss_show = loss.item()
                        
                        for i_pred in range(len(batch_mt1_cache)):
                            if i_pred == 0:
                                loss = F.cross_entropy(batch_mt1_cache[i_pred],batch_mty1_cache[i_pred])/len(batch_mt1_cache)
                            else:
                                loss = loss + F.cross_entropy(batch_mt1_cache[i_pred],batch_mty1_cache[i_pred])/len(batch_mt1_cache)
                        loss_show += loss.item()
                    
                        loss.backward()
                    
                    else:
                        loss.backward()
                        loss_show = loss.item()

                    optimizer.step()
                    if self.cfg.USE_WARM_UP:
                        with self.warmup_scheduler.dampening():
                            self.lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    sum_loss = 0.5*sum_loss + 0.5*loss_show
                    
                    y_pred.extend(pred.cpu().view(-1).detach().numpy().tolist())
                    
                    time1 = time.time()
                    print('No.',kth,'sum_loss',sum_loss,'i/all',i,len(data_in),'cost time:',time1 - time0)
                    time0 = time.time()

                    self.train_steps += batch

                    # 清空缓存
                    batch_x_cache = None
                    batch_y_cache = None
                    batch_mt1_cache = []
                    batch_mt2_cache = []
                    batch_mty1_cache = []
                    batch_mty2_cache = []

            except:
                # 清空缓存
                batch_x_cache = None
                batch_y_cache = None
                batch_mt1_cache = []
                batch_mt2_cache = []
                batch_mty1_cache = []
                batch_mty2_cache = []

                #optimizer.zero_grad()
                
                # 释放变量显存
                torch.cuda.empty_cache()

                


        # cut y_true like y_pred
        y_true = y_true[:len(y_pred)]
        
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1score = f1_score(y_true, y_pred, average='macro')

        print('sum_loss',sum_loss,'precision',precision,'recall',recall,'f1',f1score)
        print(confusion_matrix(y_true, y_pred))

        return f1score


    def mytest(self,data_in,p_text='Test'):
        self.eval() #测试模式
        
        label_dict = {} # 索引表，方便时候随机使用
        for i in range(len(data_in)):
            if data_in[i][1] not in label_dict.keys():
                label_dict[data_in[i][1]] = []
                label_dict[data_in[i][1]].append(i)
            else:
                label_dict[data_in[i][1]].append(i)
        
        #if len(label_dict.keys())<=2:
        #    print('种类不足两类，不需要进行分类....')
        #    return
        
        if self.cfg.WF_MIX_PAGE and self.cfg.DATASET_NAME == 'wf':
            bg_key = -1
            bg_size = -1
            for key in label_dict.keys():
                if len(label_dict[key]) > bg_size:
                    bg_key = key
                    bg_size = len(label_dict[key]) 
            print('bg_key',bg_key)


        test_loss = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i in range(len(data_in)):

                try:
                    # 释放变量显存
                    torch.cuda.empty_cache()
                    X = data_in[i][0]    # 一个数据
                    Y = data_in[i][1]    # 一个分类标签

                    # 复制X
                    input_X = copy.deepcopy(X)
                    if self.cfg.WF_MIX_PAGE and self.cfg.DATASET_NAME == 'wf':
                        # 加入背景流量(随机0-2)
                        for _ in range(random.randint(0,2)):
                            rand_bg_i = random.randint(0,bg_size-1)
                            input_X = self.rand_seq_keep_list_add(input_X,data_in[label_dict[bg_key][rand_bg_i]][0])

                    Label = torch.tensor([Y]).to(self.device)
                    output = self.forward(input_X,self.device)
                    test_loss += F.cross_entropy(output, Label).item()
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                    y_true.append(Y)
                    y_pred.append(pred.item())

                except:
                    continue
                    
            
        test_loss /= (1*len(data_in) )       
        
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1score = f1_score(y_true, y_pred, average='macro')

        #print('y_true',y_true,'y_pred',y_pred)

        print()
        print('Average loss',test_loss,'precision',precision,'recall',recall,'f1',f1score)
        print(confusion_matrix(y_true, y_pred))
        print()
        
        return f1score


    def start_train(self,train_data,valid_data):
        para_filename = name_generator(self.cfg)
        
        # 读取模型参数
        #model.load_state_dict(torch.load(para_filename+'.pt'))
        
        learning_rate = self.cfg.LEARNING_RATE
        best_f1_score = 0
        #开始训练
        for epoch in range(self.cfg.MAX_EPOCH):

            # 选择优化器
            if self.cfg.L2R_FLAG:
                self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay = 1e-4)
            else:
                self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

            train_f1score = self.my_train(train_data,self.optimizer,self.cfg.BATCH_SIZE,usemm=self.cfg.USEMM_FLAG)

            valid_f1score = self.mytest(valid_data,'valid')
            if valid_f1score>best_f1_score:
                best_f1_score = valid_f1score
                torch.save(self.state_dict(), 'best'+para_filename+'.pt')
            
            # 记录参数数据、训练过程
            with open(para_filename+'.txt',"a") as txt_file:
                txt_file.writelines(str(epoch)+','+str(learning_rate)+','+str(best_f1_score)+','+str(train_f1score)+','+str(valid_f1score)+'\n')

            #展示训练的模型数据
            #show_model(model,device)
        
            #保存模型
            torch.save(self.state_dict(), para_filename+'.pt')

            #学习率衰减
            learning_rate *= 0.98


        #开始测试
        #train_f1score = test(train_data,model,device,'Train')
        #valid_f1score = test(valid_data,model,device,'valid')
        #test_f1score = test(test_data,model,device)
        
        #with open(para_filename+'.txt',"a") as txt_file:
        #    txt_file.writelines(str(epoch)+','+str(learning_rate)+','+str(best_f1_score)+','+str(train_f1score)+','+str(valid_f1score)+','+str(test_f1score)+'\n')


        return 


    def start_train_k_fold(self,train_data,valid_data,test_data,kth):

        para_filename = name_generator(self.cfg) + '_' + str(kth)
        para_filename = self.cfg.OUTPATH + para_filename

        with open(para_filename+'.txt',"a") as txt_file:
            txt_file.writelines('epoch'+','+'best_valid_f1score'+','+'train_f1score'+','+'valid_f1score'+'\n')

        learning_rate = self.cfg.LEARNING_RATE
        best_f1_score = 0

        # 选择优化器
        if self.cfg.WD_FLAG:
            self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        if self.cfg.USE_WARM_UP:
            num_steps = int(len(train_data) * self.cfg.MAX_EPOCH / self.cfg.BATCH_SIZE)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_steps)
            self.warmup_scheduler = pytorch_warmup.ExponentialWarmup(self.optimizer,self.cfg.WARM_UP)

        #开始训练
        for epoch in range(self.cfg.MAX_EPOCH):
            if epoch>= self.cfg.MAX_EPOCH * 0.2:
                self.cfg.USEMM_FLAG = False

            print("epoch",epoch)
            train_f1score = self.my_train(train_data,self.optimizer,self.cfg.BATCH_SIZE,usemm=self.cfg.USEMM_FLAG,kth = kth)

            valid_f1score = self.mytest(valid_data,'valid')

            if valid_f1score>best_f1_score:
                best_f1_score = valid_f1score
                torch.save(self.state_dict(), para_filename+'_best'+'.pt')
            
            # 记录参数数据、训练过程
            with open(para_filename+'.txt',"a") as txt_file:
                txt_file.writelines(str(epoch)+','+str(best_f1_score)+','+str(train_f1score)+','+str(valid_f1score)+'\n')

            #展示训练的模型数据
            #show_model(model,device)
        
            #保存模型
            torch.save(self.state_dict(), para_filename+'.pt')

            #学习率衰减
            if not self.cfg.USE_WARM_UP:
                learning_rate *= 0.98

        
        # 读取模型参数
        if self.cfg.ES_FLAG:
            self.load_state_dict(torch.load(para_filename+'_best'+'.pt',map_location=self.device))
        else:
            self.load_state_dict(torch.load(para_filename+'.pt',map_location=self.device))

        #开始测试
        train_f1score = self.mytest(train_data,'Train')
        valid_f1score = self.mytest(valid_data,'valid')
        test_f1score = self.mytest(test_data,'Test')
        
        with open(para_filename+'.txt',"a") as txt_file:
            txt_file.writelines(str(best_f1_score)+','+str(train_f1score)+','+str(valid_f1score)+','+str(test_f1score)+'\n')

        with open(self.cfg.OUTPATH+'summary.txt',"a") as txt_file:
            txt_file.writelines(str(kth)+','+str(round(best_f1_score,5))+','+str(round(train_f1score,5))+','+str(round(valid_f1score,5))+','+str(round(test_f1score,5))+'\n')
        
        
        return 


        

