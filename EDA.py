#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 20:24:16 2020

@author: hossam
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable

import os


# import gc
# import os
# from pathlib import Path
# import random
# import sys

# from tqdm.notebook import tqdm
# import numpy as np
# import pandas as pd
# import scipy as sp


# import matplotlib.pyplot as plt
# import seaborn as sns

# from IPython.core.display import display, HTML

# # --- plotly ---
# from plotly import tools, subplots
# import plotly.offline as py
# py.init_notebook_mode(connected=True)
# import plotly.graph_objs as go
# import plotly.express as px
# import plotly.figure_factory as ff
# import plotly.io as pio
# pio.templates.default = "plotly_dark"

# # --- models ---
# from sklearn import preprocessing
# from sklearn.model_selection import KFold
# import lightgbm as lgb
# import xgboost as xgb
# import catboost as cb


# pd.set_option('max_columns', 50)

os.environ["L5KIT_DATA_FOLDER"] = "data"

cfg = load_config_data("examples/visualisation/visualisation_config.yaml")
print(cfg)


dm = LocalDataManager()
dataset_path = dm.require('scenes/sample.zarr')
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()


frames = zarr_dataset.frames
agents = zarr_dataset.agents
scenes = zarr_dataset.scenes
# tl_faces = zarr_dataset.tl_faces




# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "./"
# get config
cfg = load_config_data("../examples/visualisation_config.yaml")
print(cfg)