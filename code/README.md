## Code Structure

* **main.py** the main function 
* **config.py** the config settings
* **pytorch_warmup** Tony-Y's implementation of [pytorch_warmup](https://github.com/Tony-Y/pytorch_warmup).
* **output** the training output files
* **others** other lib py files 

## How to Config it?
All the configurations are set in [config.py](./config.py).
Here we list some of the configurations.

```
PACKET2FLOW = 'LSTM+ATT' # 'ATT, LSTM, TREE, LSTM+ATT,1dCNN, 2dCNN' 1dCNN can be used only if one flow has a fixed length of packet vectors. if 1dCNN/2dCNN, PACTETCNN should = False.   
                      # 2dCNN uses hilbertcurve2d to transfer a flow into a matrix.
FLOW2TRACE = 'LSTM+ATT' # 'ATT, LSTM, TREE, LSTM+ATT'

# the program uses k-fold
K_FOLD = 10     # k of k-fold
CORE_NUM = 1    # multi-process worker number, 1 - n

# epochs
MAX_EPOCH = 100 * 1

# paras
EMB_NUM = 100           # EMBEDDING TOKENS
F_EMB_NUM = 100         # VECTOR SIZE OF ONE EMBEDDING TOKEN
FEATURE_SIZE = 100      # HIDDEN LAYER FEATRUE SIZE
LEARNING_RATE = 0.001   # MAX LEARNING RATE
BATCH_SIZE = 16         # BATCH SIZE

# output path
OUTPATH = './output/20230505_test/'

```

Change the config file and run main.py.


