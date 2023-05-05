# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 14:46:59 2021

@author: sahua
"""


import math
import numpy as np
from PIL import Image
from hilbertcurve.hilbertcurve import HilbertCurve

'''
p=1
n=2
hilbert_curve = HilbertCurve(p, n)
distances = list(range(4**p))
points = hilbert_curve.points_from_distances(distances)

print(points)

for point, dist in zip(points, distances):
    print(f'point(h={dist}) = {point}')
'''
    
#%%
'''
img = Image.fromarray(np.zeros((50,50)))
out = img.resize((5,5))
npdata = np.array(out)
print(npdata)
'''

#%%

def fit_hilbertcurve2d(data_list,target_size,show = False):
    length = len(data_list)
    p = math.ceil(math.log(length, 4))
    p = max(p,1)
    hilbert_curve = HilbertCurve(p, 2)
    distances = list(range(4**p))
    points = hilbert_curve.points_from_distances(distances)
    
    pic_np = np.zeros((2**p,2**p))
    for i_p in range(len(points)):
        if length <= i_p:
            break
        point = points[i_p]
        pic_np[point[0],point[1]] = data_list[i_p]
    
    if show:
        print(pic_np)
    
    # 转换大小
    img = Image.fromarray(pic_np)
    out = img.resize((target_size,target_size))
    npdata = np.array(out)
    if show:
        img = Image.fromarray(pic_np*255)
        out = img.resize((target_size,target_size))
        out.show()
    
    npdata = (npdata-npdata.min())/(npdata.max()-npdata.min()+0.0000001)
    
    return npdata
    
'''
#test
x = [1,0,1,0,1,0,1,0,1,0]
pic = fit_hilbertcurve2d(x,4,show = False)
print(pic)
'''



#%%
def fit_muilti_hilbertcurve2d(data_list,target_size,show = False):
    vector_length = len(data_list[0])

    np_pic = np.zeros((vector_length,target_size,target_size))
    
    for i_v in range(vector_length):
        sub_list = list(x[i_v] for x in data_list)
        np_pic[i_v,:,:] = fit_hilbertcurve2d(sub_list,target_size,show)

    return np_pic


'''
x = [[1,0],[0,1],[1,0],[0,1]]
pic = fit_muilti_hilbertcurve2d(x,2,show = False)
print(pic)
'''




