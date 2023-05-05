# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:03:50 2222

@author: sahua
"""

import numpy as np
import torch
from model import Net  
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def draw_tsne(cfg,model_path,dataset):
    model = Net(cfg)
    model = model.to(cfg.DEVICE)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

    feature_list = []
    Y_list = []
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            print(i,len(dataset))
            X = dataset[i][0]    # 一个数据
            Y = dataset[i][1]    # 一个分类标签
            output = model.get_last_layer(X)[0]
            output = output.cpu().numpy()
            feature_list.append(output)
            Y_list.append(Y)
    
    # 计算tsne 降维
    tsne = TSNE(n_components=2).fit_transform(feature_list)

    # 进行缩放
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # 开始画图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    plt.xticks(fontproperties = 'Times New Roman')
    plt.yticks(fontproperties = 'Times New Roman')
    
    
    #colors_per_class = cm.rainbow(np.linspace(0, 1, cfg.CLASS_NUM))
    #random.shuffle(colors_per_class)
    colors_per_class = get_cmap(cfg.CLASS_NUM + 1)
    
    
    for label in range(cfg.CLASS_NUM):
        # find the samples of the current class in the data
        indices = [i for i in range(len(Y_list)) if Y_list[i] == label]
        
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format

        color = colors_per_class(label+1)
        print('len(current_tx)',len(current_tx))
        color_repeat = [color for _ in range(len(current_tx))]
        
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=color_repeat, label=label)
        
        
    #ax.legend(loc='best')
    
    plt.savefig('tsen.pdf')
    plt.show()
    
    return 







