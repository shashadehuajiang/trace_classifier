# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 19:29:59 2021

@author: sahua
"""

import os
#from scapy.all import *
import time
import json

import dpkt
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
                    
                    #print(filename2)




def fast_pkt_info(buf,ts):
    eth = dpkt.ethernet.Ethernet(buf)  # 解包，物理层
 
    if not isinstance(eth.data, dpkt.ip.IP):  # 解包，网络层，判断网络层是否存在
        return []
 
    ip = eth.data
 
    src = socket.inet_ntoa(ip.src)  # 源地址
    dst = socket.inet_ntoa(ip.dst)  # 目的地址
 
    transf_type = 0
    if isinstance(ip.data, dpkt.udp.UDP):  # 解包，判断传输层协议是否是UDP
        transf_type = 17
        transf_data = ip.data
        sport = transf_data.sport  # 源端口
        dport = transf_data.dport  # 目的端口
    elif isinstance(ip.data, dpkt.tcp.TCP):
        transf_type = 6
        transf_data = ip.data
        sport = transf_data.sport  # 源端口
        dport = transf_data.dport  # 目的端口
    else:
        return []
    
    if src > dst:
        direction = 1
        unit = (transf_type,src,dst,sport,dport)
    else:
        direction = -1
        unit = (transf_type,dst,src,dport,sport)
    
    data = [round(ts, 10), int(direction*len(buf)), transf_type]
    
    return [unit,data]
    


def fast_read_pcap(input_file,label,split_second = 5000):
    return_list = []
    flows = {}
    ts = 0
    
    with open(input_file,'rb') as f:
        try:
            pcap = dpkt.pcap.Reader(f)  # 先按.pcap格式解析，若解析不了，则按pcapng格式解析
        except:
            f.seek(0,0)
            pcap = dpkt.pcapng.Reader(f)
        
        first_start_flag = True
        start_time = 0
            
        pkt_id = 0  # 初始序列为0
        # 将时间戳和包数据分开，一层一层解析，其中ts是时间戳，buf存放对应的包
        for (ts, pkt) in pcap:  # 遍历pcap数据
            pkt_id += 1
            if first_start_flag == True:
                first_start_flag = False
                start_time = ts
                pktinfo1 = fast_pkt_info(pkt,ts)
                if len(pktinfo1) >0 :
                    unit = pktinfo1[0]
                    data = pktinfo1[1]
                    if unit not in flows.keys():
                        flows[unit] = []
                    flows[unit].append(data)
                continue
            
            if ts - start_time <= split_second:
                #print(pkt)
                pktinfo1 = fast_pkt_info(pkt,ts)
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
                start_time = ts
                flows = {}
                  
    list_values = [i for i in flows.values()]
    return_list.append([list_values,label])
    
    #后处理
    start_time = ts
    flows = {}

    return return_list



def main():
    walkFile("./pcaps/")
    print('len(file_path_dcit.keys())',len(file_path_dcit.keys()))
    
    label = 0
    for key in file_path_dcit.keys():
        sample_num =0 
        for pcapfile in file_path_dcit[key]:
            if not (pcapfile.endswith('.pcap') or pcapfile.endswith('.pcapng') ):
                continue
            
            print('label',label,'sample_num',sample_num)
            print(pcapfile)
            #try:
            return_list = fast_read_pcap(pcapfile,label)
            #except:
            #    return_list = []
            print(len(return_list))
                
            sample_num += len(return_list)
            final_labeled_list.extend(return_list)
        
        print(key,'sample_num',sample_num)
        
            
        label += 1
        label2key.append(key)
            
    
    print('Now saving...')
    # 保存到json文件
    content = [final_labeled_list,label2key]
    filename = "pcap.json"
    with open(filename, 'w') as file_obj:
        json.dump(content, file_obj)
    
    

if __name__ == '__main__':
    main()
    
    