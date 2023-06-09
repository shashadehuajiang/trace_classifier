# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 19:29:59 2021

@author: sahua
"""

import os
from scapy.all import *
import time
import json

#import dpkt
import socket

#%%

file_path_dcit = {}
final_labeled_list = []
label2key = []


# 遍历文件夹
def walkFile(file):
    for root, dirs, files in os.walk(file):
        # 遍历所有的文件夹
        for d in dirs:
            filename = os.path.join(root, d)

            file_path_dcit[d] = []
            for root2, dirs2, files2 in os.walk(filename):
                for d2 in files2:
                    filename2 = os.path.join(root2, d2)
                    file_path_dcit[d].append(filename2)


def fast_pkt_info(i_pkt):
    try:
        if i_pkt['IP'].proto == 6: #TCP
            src = i_pkt['IP'].src
            dst = i_pkt['IP'].dst
            sport = i_pkt['TCP'].sport
            dport = i_pkt['TCP'].dport
            
        elif  i_pkt['IP'].proto == 17: #UDP
            src = i_pkt['IP'].src
            dst = i_pkt['IP'].dst
            sport = i_pkt['UDP'].sport
            dport = i_pkt['UDP'].dport
        else:
            return []
        
        transf_type = i_pkt['IP'].proto
        
        # 这边因为src都相等所以可以用这种方式，否则需要自己输入src处理一下方向。
        if src > dst:
            direction = 1
            unit = (transf_type,src,dst,sport,dport)
        else:
            direction = -1
            unit = (transf_type,dst,src,dport,sport)
            
        data = [float(i_pkt.time), int(direction*len(i_pkt)), transf_type]
        return [unit,data]
        
    except:
        return []
        

    


def fast_read_pcap(input_file,label,split_second = 5000):
    return_list = []
    flows = {}
    
    first_start_flag = True
    start_time = 0
    pkt_id = 0  # 初始序列为0
    
    with PcapReader(input_file) as pcap_reader:

        for pkt in pcap_reader:  # 遍历pcap数据
            pkt_id += 1
            if pkt_id%10000 == 0:
                print(pkt_id)

            if first_start_flag == True:
                first_start_flag = False
                start_time = pkt.time
                pktinfo1 = fast_pkt_info(pkt)
                if len(pktinfo1) >0 :
                    unit = pktinfo1[0]
                    data = pktinfo1[1]
                    if unit not in flows.keys():
                        flows[unit] = []
                    flows[unit].append(data)
                continue
            
            if pkt.time - start_time <= split_second:
                pktinfo1 = fast_pkt_info(pkt)
                if len(pktinfo1) >0 :
                    unit = pktinfo1[0]
                    data = pktinfo1[1]
                    if unit not in flows.keys():
                        flows[unit] = []
                    flows[unit].append(data)
            else:
                # 进行分流，处理特征
                list_values = [i for i in flows.values()]
                return_list.append([list_values,label])
                
                #后处理
                start_time = start_time + split_second
                flows = {}
                  
                

    return return_list



def main():
    walkFile("./pcaps/")
    print(file_path_dcit)
    label = 0
    for key in file_path_dcit.keys():
        sample_num =0 
        for pcapfile in file_path_dcit[key]:
            if not (pcapfile.endswith('.pcap') or pcapfile.endswith('.pcapng') ):
                continue
            print(pcapfile)
            return_list = fast_read_pcap(pcapfile,label)
            print(len(return_list))
            sample_num += len(return_list)
            final_labeled_list.extend(return_list)
        
        print(key,'sample_num',sample_num)
        
            
        label += 1
        label2key.append(key)
            
    
    print('Now save file...')
    # 保存到json文件
    content = [final_labeled_list,label2key]
    filename = "pcap.json"
    with open(filename, 'w') as file_obj:
        json.dump(content, file_obj)
    
    

if __name__ == '__main__':
    main()
    