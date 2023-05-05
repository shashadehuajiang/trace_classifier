# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:03:50 2020

@author: sahua
"""

import random
import numpy as np
import copy
import csv

from my_hilbertcurve import fit_muilti_hilbertcurve2d
from scipy import interpolate



def dataset_use_1dscale(dataset,size):
    data = []
    for i in range(len(dataset)):
        if i%10 == 0:
            print(i,len(dataset))

        new_trace = copy.deepcopy(dataset[i][0])
        for i_flow in range(len(new_trace)):
            vector_length = len(new_trace[i_flow][0])
            sacled_vector_list = []

            for i_v in range(vector_length):
                sub_list = list(x[i_v] for x in new_trace[i_flow])
                if len(sub_list) == 1:
                    sub_list.extend(sub_list)

                x_array = np.array(list(range(len(sub_list))))/ (len(sub_list)-1)
                #print('x_array',x_array)
                #print('sub_list',sub_list)
                x_new = np.array(list(range(size))) / size
                f = interpolate.interp1d(x_array,sub_list)
                ynew = f(x_new)

                sacled_vector_list.append(ynew)

            new_flow = []
            sacled_vector_list_array = np.array(sacled_vector_list)
            for i_v in range(sacled_vector_list_array.shape[1]):
                vector = sacled_vector_list_array[:,i_v].tolist()
                new_flow.append([vector])
                #print(vector)

            new_trace[i_flow] = new_flow

        data.append([new_trace,dataset[i][1]])

    return data



def dataset_use_hilbertcurve(dataset,size):

    data = []
    for i in range(len(dataset)):

        if i%10 == 0:
            print(i,len(dataset))
        
        new_data = (dataset[i][0])
        
        for i_flow in range(len(new_data)):
            new_data[i_flow] = fit_muilti_hilbertcurve2d(new_data[i_flow],size)
        
        data.append([new_data,dataset[i][1]])

    return data


def dataset2flow(dataset):
    new_dataset = []
    for i in range(len(dataset)):
        for i_f in range(len(dataset[i][0])):
            oneflow = dataset[i][0][i_f]
            new_data = [[oneflow],dataset[i][1]]
            new_dataset.append(new_data)

    return new_dataset



def tailoring(flow, fsize = 10):
    new_flow = copy.deepcopy(flow)
    if len(flow)>fsize:
        new_flow = new_flow[:fsize]
    
    if len(flow)<fsize:
        # 建立全0向量
        zero_vector = []
        for i in flow[0]:
            zero_vector.append(0)
        
        for i in range(fsize-len(flow)):
            new_flow.append(zero_vector)
    
    return new_flow


def tailoring_dataset(data,fsize = 10):
    print('start tailoring_dataset...   flow size:',fsize)
    for i in range(len(data)):
        for i_f in range(len(data[i][0])):
            data[i][0][i_f] = tailoring(data[i][0][i_f],fsize)
    print('finish tailoring_dataset...')
    return data



def drop_empty_data(dataset):
    for i in range(len(dataset)-1,-1,-1):
        if len(dataset[i][0]) == 0:
            del dataset[i]
    return dataset


def  pre_process_1(dataset, x_norm = None):
    # 添加绝对时间
    for i in range(len(dataset)):
        for i_f in range(len(dataset[i][0])):
            for i_v in range(len(dataset[i][0][i_f])):
                if i_v>=1:
                    time_sum = dataset[i][0][i_f][i_v][0] + dataset[i][0][i_f][i_v-1][-1]
                    dataset[i][0][i_f][i_v] = np.r_[dataset[i][0][i_f][i_v],time_sum]
                else:
                    dataset[i][0][i_f][i_v] = np.r_[dataset[i][0][i_f][i_v],0]
    print('add percentage3 finish...')

    if x_norm is None:

        #数据大小归一化
        def min_np(a,b):
            return_x = copy.deepcopy(a)
            for i in range(len(return_x)):
                return_x[i] = min(return_x[i],b[i])
            return return_x

        def max_np(a,b):
            return_x = copy.deepcopy(a)
            for i in range(len(return_x)):
                return_x[i] = max(return_x[i],b[i])
            return return_x

        packet_dim = len(dataset[i][0][0][0])
        print('packet_dim:',packet_dim)
        min_a = np.zeros(packet_dim)
        max_a = np.zeros(packet_dim)

        for i in range(len(dataset)):
            for i_vectors in dataset[i][0]:
                for i_v in i_vectors:
                    min_a = min_np(min_a,i_v)
                    max_a = max_np(max_a,i_v)

        x_norm = [min_a,max_a]
    
    min_a = x_norm[0]
    max_a = x_norm[1]

    for i in range(len(dataset)):
        for i_vectors in range(len(dataset[i][0])):
            for i_v in range(len(dataset[i][0][i_vectors])):
                dataset[i][0][i_vectors][i_v] = (dataset[i][0][i_vectors][i_v]-min_a)/(max_a-min_a)



    # 添加顺序，并clip
    for i in range(len(dataset)):
        for i_vectors in range(len(dataset[i][0])):
            for i_v in range(len(dataset[i][0][i_vectors])):
                id_pad = min(i_v/100,1)
                dataset[i][0][i_vectors][i_v] = np.r_[dataset[i][0][i_vectors][i_v],id_pad]


    print('add percentage finish...')

    # 添加百分比
    for i in range(len(dataset)):
        for i_vectors in range(len(dataset[i][0])):
            for i_v in range(len(dataset[i][0][i_vectors])):
                percentage = i_v/len(dataset[i][0][i_vectors])
                dataset[i][0][i_vectors][i_v] = np.r_[dataset[i][0][i_vectors][i_v],percentage]

    print('add percentage2 finish...')

    return dataset, x_norm


def  pre_process(train_data, valid_data, test_data):
    drop_empty_data(train_data)
    drop_empty_data(valid_data)
    drop_empty_data(test_data)
    
    #数据大小归一化
    def min_np(a,b):
        return_x = copy.deepcopy(a)
        for i in range(len(return_x)):
            return_x[i] = min(return_x[i],b[i])
        return return_x

    def max_np(a,b):
        return_x = copy.deepcopy(a)
        for i in range(len(return_x)):
            return_x[i] = max(return_x[i],b[i])
        return return_x

    min_a = np.zeros(2)
    max_a = np.zeros(2)
    for i in range(len(train_data)):
        for i_vectors in train_data[i][0]:
            for i_v in i_vectors:
                min_a = min_np(min_a,i_v)
                max_a = max_np(max_a,i_v)

    for i in range(len(train_data)):
        for i_vectors in range(len(train_data[i][0])):
            for i_v in range(len(train_data[i][0][i_vectors])):
                train_data[i][0][i_vectors][i_v] = (train_data[i][0][i_vectors][i_v]-min_a)/(max_a-min_a)

    for i in range(len(valid_data)):
        for i_vectors in range(len(valid_data[i][0])):
            for i_v in range(len(valid_data[i][0][i_vectors])):
                valid_data[i][0][i_vectors][i_v] = (valid_data[i][0][i_vectors][i_v]-min_a)/(max_a-min_a)
        
    for i in range(len(test_data)):
        for i_vectors in range(len(test_data[i][0])):
            for i_v in range(len(test_data[i][0][i_vectors])):
                test_data[i][0][i_vectors][i_v] = (test_data[i][0][i_vectors][i_v]-min_a)/(max_a-min_a)

    #print('test_data[0][0][0][0]',test_data[0][0][0][0])



    # 添加顺序，并clip
    for i in range(len(train_data)):
        for i_vectors in range(len(train_data[i][0])):
            for i_v in range(len(train_data[i][0][i_vectors])):
                id_pad = min(i_v/100,1)
                train_data[i][0][i_vectors][i_v] = np.r_[train_data[i][0][i_vectors][i_v],id_pad]

    for i in range(len(valid_data)):
        for i_vectors in range(len(valid_data[i][0])):
            for i_v in range(len(valid_data[i][0][i_vectors])):
                id_pad = min(i_v/100,1)
                valid_data[i][0][i_vectors][i_v] = np.r_[valid_data[i][0][i_vectors][i_v],id_pad]

    for i in range(len(test_data)):
        for i_vectors in range(len(test_data[i][0])):
            for i_v in range(len(test_data[i][0][i_vectors])):
                id_pad = min(i_v/100,1)
                test_data[i][0][i_vectors][i_v] = np.r_[test_data[i][0][i_vectors][i_v],id_pad]

    print('add percentage2 finish...')


    # 添加百分比
    for i in range(len(train_data)):
        for i_vectors in range(len(train_data[i][0])):
            for i_v in range(len(train_data[i][0][i_vectors])):
                percentage = i_v/len(train_data[i][0][i_vectors])
                train_data[i][0][i_vectors][i_v] = np.r_[train_data[i][0][i_vectors][i_v],percentage]

    for i in range(len(valid_data)):
        for i_vectors in range(len(valid_data[i][0])):
            for i_v in range(len(valid_data[i][0][i_vectors])):
                percentage = i_v/len(valid_data[i][0][i_vectors])
                valid_data[i][0][i_vectors][i_v] = np.r_[valid_data[i][0][i_vectors][i_v],percentage]

    for i in range(len(test_data)):
        for i_vectors in range(len(test_data[i][0])):
            for i_v in range(len(test_data[i][0][i_vectors])):
                percentage = i_v/len(test_data[i][0][i_vectors])
                test_data[i][0][i_vectors][i_v] = np.r_[test_data[i][0][i_vectors][i_v],percentage]

    print('add percentage finish...')


def cut_dataset(data_in,val_rate = 0.1, test_rate = 0.1):
    
    label_dict = {} # 索引表，方便时候随机使用
    for i in range(len(data_in)):
        if data_in[i][1] not in label_dict.keys():
            label_dict[data_in[i][1]] = []
            label_dict[data_in[i][1]].append(i)
        else:
            label_dict[data_in[i][1]].append(i)

    train_data = []
    valid_data = []
    test_data = []
    for key in label_dict.keys():
        value_list = label_dict[key]
        sub_data_list = [data_in[i] for i in value_list]
        random.seed(1)
        random.shuffle(sub_data_list)

        test_set_size = int(len(sub_data_list)*test_rate)
        valid_set_size = int(len(sub_data_list)*val_rate)
        train_set_size = len(sub_data_list) - test_set_size - valid_set_size

        sub_train_data = sub_data_list[:train_set_size]
        sub_valid_data = sub_data_list[train_set_size:train_set_size+valid_set_size]
        sub_test_data = sub_data_list[train_set_size+valid_set_size:]

        train_data.extend(sub_train_data)
        valid_data.extend(sub_valid_data)
        test_data.extend(sub_test_data)

    return train_data,valid_data,test_data


def cut_dataset2(dataset,val_rate = 0.1, test_rate = 0.1):
    
    test_set_size= int(len(dataset)*test_rate)
    valid_set_size = int(len(dataset)*val_rate)
    train_set_size = len(dataset) - test_set_size - valid_set_size
    train_data = dataset[:train_set_size]
    valid_data = dataset[train_set_size:train_set_size+valid_set_size]
    test_data = dataset[train_set_size+valid_set_size:]

    return train_data,valid_data,test_data



def name_generator(cfg):
    a1 = 'T' if cfg.PACTETCNN else 'F'
    a1_2 = 'T' if cfg.TRANSFORMER else 'F'
    a2 = cfg.PACKET2FLOW
    a3 = cfg.FLOW2TRACE
    
    dataset = cfg.DATASET_NAME

    b0 = 'T' if cfg.LNORM_FLAG else 'F'
    b1 = 'T' if cfg.WD_FLAG else 'F'
    b2 = 'T' if cfg.DROPOUT_FLAG else 'F'
    b3 = 'T' if cfg.USEBN_FALG else 'F'
    b4 = 'T' if cfg.USEMM_FLAG else 'F'
    
    filename = 'model_paras_' +  a1 + a1_2 + '_'+ a2 + a3 + '_' + dataset + '_' + b0+ b1 + b2 + b3 + b4 

    return filename




def process_final_results(cfg):

    data_list = []
    with open(cfg.OUTPATH+'summary.txt','r')as f:
        f_csv = csv.reader(f)
        for line in f_csv:
            print(line)
            if len(line) == 5:
                data_list.append(float(line[4]))

    data_array = np.array(data_list)
    m = np.mean(data_array)
    std = np.std(data_array)
    
    print('mean:',m,'std:',std)

    with open(cfg.OUTPATH+'summary_output.txt',"a") as txt_file:
        txt_file.writelines('mean:'+ str(m) + ' std:'+str(std) + '\n')

    return 






