# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:03:50 2020

@author: sahua
"""


import random
import json
import config

def dataset_json(json_name = '', rand_seed = 1, twoclass = False):
    data_path = config.DATASET_FOLDER + json_name +  ".json"

    print('start to readfile...')
    with open(data_path,'r') as json_file:
        content = json.load(json_file)
        data = content[0]
        label2key = content[1]

    random.seed(1)
    random.shuffle(data)

    data_cha = data
    for i in range(len(data_cha)):
        if len(data[i][0])>0:
            start_time = data[i][0][0][0][0]
        for i_f in range(len(data_cha[i][0])):
            for i_v in range(len(data_cha[i][0][i_f])-1,-1,-1):
                if i_v>=1:
                    data_cha[i][0][i_f][i_v][0] -= data[i][0][i_f][i_v-1][0]
                elif i_v==0:
                    data_cha[i][0][i_f][i_v][0] -= start_time

    print('read dataset finish...')
    print('label2key',label2key)

    return data_cha, label2key


