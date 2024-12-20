# =============== OS, System, and Environment Variables =================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Restrict to GPU 1

# =============== Data Handling and Preprocessing =======================
import pandas as pd
import numpy as np

# =============== PyTorch Imports =======================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter

# =============== Torchaudio Imports ====================================
import torchaudio
from torchaudio import transforms as T
from torchaudio.transforms import TimeMasking, FrequencyMasking, Resample

# =============== Transformers and Audio Models =========================
from transformers import (
    AutoFeatureExtractor, 
    Data2VecAudioModel, 
    Wav2Vec2Processor, 
    Wav2Vec2Model, 
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    AutoProcessor, 
    HubertModel, 
    Wav2Vec2FeatureExtractor, 
    UniSpeechModel, 
    ASTForAudioClassification, 
    ASTFeatureExtractor, 
    WhisperModel
)

# =============== Scikit-Learn Imports ===================================
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.metrics import log_loss, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# =============== Machine Learning Libraries ============================
import lightgbm as lgb
import CatBoost_code as cb
import xgboost as xgb
from CatBoost_code import CatBoostClassifier, Pool

# =============== Image and Visualization Libraries ======================
import matplotlib.pyplot as plt

# =============== Other Utility Libraries ================================
import tqdm
import math
import random
from typing import Any

# =============== Specialized Libraries ===================================
import whisper
from efficientnet_pytorch import EfficientNet
