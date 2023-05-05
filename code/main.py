# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:03:50 2020

@author: sahua
"""

import os
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process, Manager

from dataset import dataset_json
from model import Net  
from mylib import cut_dataset,pre_process,pre_process_1,process_final_results,tailoring_dataset,dataset2flow,dataset_use_hilbertcurve,dataset_use_1dscale,drop_empty_data
from config import G_CONFIG
import torch
import copy

from draw_tsne import draw_tsne
from sklearn.utils import check_random_state


def read_dataset(cfg):
    dataset, label2key = dataset_json(cfg.DATASET_NAME)
    cfg.CLASS_NUM = len(label2key)
    cfg.PACKTE_DIM = len(dataset[0][0][0][0]) + 3
    print("cfg.PACKTE_DIM",cfg.PACKTE_DIM)
   
    print('start dataset2flow()...')
    if cfg.HU2_FLAG:
        dataset = dataset2flow(dataset)

    return dataset 

def one_of_k_fold_test(cfg, dataset, ith):
    print('No.',ith,'start...')

    if dataset is None:
        print('No.',ith,'read.. dataset')
        dataset = read_dataset(cfg)
        print('No.',ith,'start pre_precess...')
        if cfg.FLOW_CUT_FALG:
            dataset = tailoring_dataset(dataset,cfg.FLOW_CUT_SIZE)
        drop_empty_data(dataset)


        if cfg.PACKET2FLOW == '2dCNN':
            dataset = dataset_use_hilbertcurve(dataset,cfg.SIZE_2dCNN)

        if cfg.SCALE_1dCNN == True and cfg.PACKET2FLOW == '1dCNN':
            dataset = dataset_use_1dscale(dataset,cfg.SCALE_SIZE)

    n_fold = cfg.K_FOLD
    cutpoint = int(ith * len(dataset)/ n_fold)
    mydata = dataset[cutpoint: ] + dataset[:cutpoint]

    print('No.',ith,'start cut_dataset...')
    train_data,valid_data,test_data = cut_dataset(mydata,0.1,0.1)
    train_data = train_data[:min(cfg.TRAIN_SAMPLE_MAX,len(train_data))]
    valid_data = valid_data[:min(cfg.VALID_SAMPLE_MAX,len(valid_data))]

    print('No.',ith,'cut finish...')

    train_data, x_norm = pre_process_1(train_data, x_norm = None)
    valid_data, _ = pre_process_1(valid_data, x_norm = x_norm)
    test_data, _ = pre_process_1(test_data, x_norm = x_norm)
    
    print("len(train_data)",len(train_data))
    print("len(valid_data)",len(valid_data))
    print("len(test_data)",len(test_data))

    model = Net(cfg)
    model = model.to(cfg.DEVICE)
    model.start_train_k_fold(train_data,valid_data,test_data,ith)

    return 

def print_error(value):
    print("error: ", value)


def one_fold_test(cfg):
    os.makedirs(cfg.OUTPATH,exist_ok=True)

    dataset = read_dataset(cfg)

    print('start pre_precess...')
    if cfg.FLOW_CUT_FALG:
        dataset = tailoring_dataset(dataset,cfg.FLOW_CUT_SIZE)
    drop_empty_data(dataset)

    
    if cfg.PACKET2FLOW == '2dCNN':
        dataset = dataset_use_hilbertcurve(dataset,cfg.SIZE_2dCNN)

    if cfg.SCALE_1dCNN == True and cfg.PACKET2FLOW == '1dCNN':
        dataset = dataset_use_1dscale(dataset,cfg.SCALE_SIZE)
    
    import time
    time0 = time.time()
    one_of_k_fold_test(cfg, dataset, 0)
    time1 = time.time()
    print('cost time',time1 - time0)
    
    return 


def k_fold_test(cfg):
    if cfg.DATASET_SHARE:
        print('this operation is not ready yet!')
        os.exit()

        dataset = read_dataset(cfg)

        print('start pre_precess...')
        if cfg.FLOW_CUT_FALG:
            dataset = tailoring_dataset(dataset,cfg.FLOW_CUT_SIZE)
        drop_empty_data(dataset)


        if cfg.PACKET2FLOW == '2dCNN':
            dataset = dataset_use_hilbertcurve(dataset,cfg.SIZE_2dCNN)
        if cfg.SCALE_1dCNN == True and cfg.PACKET2FLOW == '1dCNN':
            dataset = dataset_use_1dscale(dataset,cfg.SCALE_SIZE)

        print('Warning: code not ready yet... Unable to use a share space')
        manager = Manager()
        share_list = manager.list()
        for i in range(len(dataset)):
            share_list.append(dataset[i])
            if i %100 == 0:
                print(i,len(dataset))
        print('finish...')
    else:
        share_list = None

    print('mkdir output filepath...')
    os.makedirs(cfg.OUTPATH,exist_ok=True)

    print('build process pool...')
    pool = Pool(processes=cfg.CORE_NUM)
    for i in range(0, cfg.K_FOLD):
        pool.apply_async(one_of_k_fold_test, (cfg, share_list, i), error_callback=print_error)
    
    print('build finish...')
    
    pool.close()
    pool.join()

    return 

def one_round_test(cfg):
    dataset = read_dataset(cfg)
    train_data,valid_data,test_data = cut_dataset(dataset)
    pre_process(train_data,valid_data,test_data)

    model = Net(cfg)
    model = model.to(cfg.DEVICE)
    model.start_train(train_data,valid_data)

    return 


def main():
    
    torch.multiprocessing.set_start_method('spawn')

    cfg = G_CONFIG()

    #one_fold_test(cfg)

    k_fold_test(cfg)
    process_final_results(cfg)

    '''
    dataset = read_dataset(cfg)
    print('start pre_precess...')
    pre_process_1(dataset)
    model_path = './output/tsen/ISD.pt'
    draw_tsne(cfg,model_path,dataset)
    '''

    return 


if __name__ == '__main__':
    main()



    
    
    
    


