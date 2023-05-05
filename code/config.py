# -*- coding: utf-8 -*-
import torch

DATASET_SHARE = False # is not ready...

PACTETCNN = True
TRANSFORMER = False

PACKET2FLOW = 'LSTM+ATT' # 'ATT, LSTM, TREE, LSTM+ATT,1dCNN, 2dCNN' 1dCNN can be used only if one flow has a fixed length of packet vectors. if 1dCNN/2dCNN, PACTETCNN should = False.   
                      # 2dCNN uses hilbertcurve2d to transfer a flow into a matrix.
FLOW2TRACE = 'LSTM+ATT' # 'ATT, LSTM, TREE, LSTM+ATT'

# k-fold
K_FOLD = 10
CORE_NUM = 1    # use worker number, 1 - n

# epochs
MAX_EPOCH = 100 * 1

# gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATASET
DATASET_FOLDER = '../dataset/'
DATASET_NAME = 'test' 

# handle overfitting
ES_FLAG = True
LNORM_FLAG = True
WD_FLAG = True
DROPOUT_FLAG = False
USEBN_FALG = True
USEMM_FLAG = True
DATA_ENHANCEMENT_FLAG = True

# paras
CLASS_NUM = -1 # auto set
PACKTE_DIM = -1 # auto set
EMB_NUM = 100
F_EMB_NUM = 100
FEATURE_SIZE = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 16 # half batch

USE_WARM_UP = True
WARM_UP = int(2000/BATCH_SIZE) # 0 - n 

# output
OUTPATH = './output/20230505_test/'

# expriments
FLOW_CUT_FALG = False
FLOW_CUT_SIZE = 2

# expriments
SCALE_1dCNN = False # 进行重采样    if PACKET2FLOW == 1dCNN
SCALE_SIZE = 16*16

# expriments
HU2_FLAG = False # 按照flow进行分类, HU2
FIX_M1_CNN_LENGTH =  SCALE_SIZE # FLOW_CUT_SIZE # if PACKET2FLOW == 1dCNN
SIZE_2dCNN = 16
TRAIN_SAMPLE_MAX = 1e100
VALID_SAMPLE_MAX = 1e100

# expriments
WF_MIX_PAGE = False

class G_CONFIG:
    def __init__(self):
        self.DATASET_SHARE = DATASET_SHARE
        self.PACTETCNN = PACTETCNN
        self.TRANSFORMER = TRANSFORMER
        self.USE_WARM_UP = USE_WARM_UP
        self.WARM_UP = WARM_UP
        self.PACKET2FLOW = PACKET2FLOW
        self.FLOW2TRACE = FLOW2TRACE
        self.K_FOLD = K_FOLD
        self.CORE_NUM = CORE_NUM
        self.MAX_EPOCH = MAX_EPOCH
        self.DEVICE = DEVICE
        self.DATASET_FOLDER = DATASET_FOLDER
        self.DATASET_NAME = DATASET_NAME
        self.ES_FLAG = ES_FLAG
        self.LNORM_FLAG = LNORM_FLAG
        self.WD_FLAG = WD_FLAG
        self.DROPOUT_FLAG = DROPOUT_FLAG
        self.USEBN_FALG = USEBN_FALG
        self.USEMM_FLAG = USEMM_FLAG
        self.DATA_ENHANCEMENT_FLAG = DATA_ENHANCEMENT_FLAG
        self.CLASS_NUM = CLASS_NUM
        self.PACKTE_DIM = PACKTE_DIM
        self.EMB_NUM = EMB_NUM
        self.F_EMB_NUM = F_EMB_NUM
        self.FEATURE_SIZE = FEATURE_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.FLOW_CUT_FALG = FLOW_CUT_FALG
        self.FLOW_CUT_SIZE = FLOW_CUT_SIZE
        self.HU2_FLAG = HU2_FLAG
        self.FIX_M1_CNN_LENGTH = FIX_M1_CNN_LENGTH
        self.SCALE_1dCNN = SCALE_1dCNN
        self.SCALE_SIZE = SCALE_SIZE
        self.SIZE_2dCNN = SIZE_2dCNN
        self.TRAIN_SAMPLE_MAX = TRAIN_SAMPLE_MAX
        self.VALID_SAMPLE_MAX = VALID_SAMPLE_MAX
        self.WF_MIX_PAGE = WF_MIX_PAGE
        self.OUTPATH = OUTPATH























