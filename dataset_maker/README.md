# Intro
--------
It is a small tool to help you build dataset .json file from the pcap files.

## Dataset file structure

The pcap dataset need to be organized like the following structure:

- pcaps

-- Label A
  ---1.pcap
  ---2.pcap
  ---3.pcap

-- Label B
  ---1.pcap
  ---2.pcap
  ---3.pcap

-- Label C
  ---1.pcap
  ---2.pcap
  ---3.pcap


## RUN

```
python3  pcap2json_dpkt.py #(or run the other two python files)

# pcap2json_dpkt.py can be the default choice. It uses dpkt to process the files.
# pcap2json_dpkt_plus_payload.py add the payload bytes to the packet vectors. 
# pcap2json_scapy.py uses scapy lib to process the files. 
```

## Note

1. dpkt is fast. But it may make errors on some packets, and the reason is unknown. Then you may try scapy.

2. "split_second" in the python files means the time window size of one sample. You can set it to other values.

3. There is no real pcap in the './pcaps/' yet. So do not run the python files before you replace the pcaps.


