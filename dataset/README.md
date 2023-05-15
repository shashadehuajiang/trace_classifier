# trace_classifier
Here we list the datasets used. They can be found at [Kaggle](https://www.kaggle.com/datasets/fuukaa/network-multiflow-fingerprinting-datasets?datasetId=3270419).

* User Activities (UAV). 

** 5pmnkshffm_5s_wfilter.json

** [Labayen's dataset](https://data.mendeley.com/datasets/5pmnkshffm/3)

* IoT Device Identification (IDI).

** IoT_Sentinel_plus_payload500.json

** [Miettinen's dataset](https://github.com/andypitcher/IoT_Sentinel)

* Intrusion Detection (ISD)

** ETF_IoT.json

** [Jovanovic's dataset](https://data.mendeley.com/datasets/nbs66kvx6n)

* Shadowsocks Website Fingerprinting (SWF).

** wf_dataset.json

** The original PCAP files are missing hard drive corruption.

* Keyword Searching (KWS).

** android_search50.json

** Keyword Search Dataset Pcap.zip provides the origal pcaps.

After downloading these json files, copy them to the ./dataset path, and then change the config.py.
You can run these datasets now.

## dataset format
generate_a_dataset(sample).py shows how to generate a fake dataset.

Here is how one dataset looks like.

dataset = [sample1, sample2, sample3, ..., sample n]

sample = [[flow1, flow2, flow3, ..., flow n], label]

flow1 = [[packet vector1, packet vector2, packet vector3, ...,packet vector n]]

Make sure that all the packet vectors have the same dimensions.
