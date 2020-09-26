#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 20:24:16 2020

@author: hossam
"""
import matplotlib.pyplot as plt
from matplotlib import animation, rc
rc('animation', html='jshtml')
from IPython.display import HTML
#from IPython.display import display, clear_output
import PIL

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

from IPython.display import display, clear_output

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

# Loading sample data for EDA
# set env variable for data

dm = LocalDataManager()
dataset_path = dm.require('scenes/sample.zarr')
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
print(zarr_dataset)



frames = zarr_dataset.frames
agents = zarr_dataset.agents
scenes = zarr_dataset.scenes
# tl_faces = zarr_dataset.tl_faces




# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "data"
# get config
cfg = load_config_data("examples/visualisation/visualisation_config.yaml")
print(cfg)







# Visualization Functions

def animate_solution(images, timestamps=None):
    def animate(i):
        changed_artifacts = [im]
        im.set_data(images[i])
        if timestamps is not None:
            time_text.set_text(timestamps[i])
            changed_artifacts.append(im)
        return tuple(changed_artifacts)


    fig, ax = plt.subplots()
    im = ax.imshow(images[0])
    if timestamps is not None:
        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    anim = animation.FuncAnimation(fig, animate, frames=len(images), interval=60, blit=True)

    # To prevent plotting image inline.
    plt.close()
    return anim

def visualize_rgb_image(dataset, index, title="", ax=None):
    """Visualizes Rasterizer's RGB image"""
    data = dataset[index]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)

    if ax is None:
        fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    ax.imshow(im[::-1])

def create_animate_for_indexes(dataset, indexes):
    images = []
    timestamps = []

    for idx in indexes:
        data = dataset[idx]
        im = data["image"].transpose(1, 2, 0)
        im = dataset.rasterizer.to_rgb(im)
        target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
        center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
        draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)
        clear_output(wait=True)
        images.append(PIL.Image.fromarray(im[::-1]))
        timestamps.append(data["timestamp"])

    anim = animate_solution(images, timestamps)
    return anim

def create_animate_for_scene(dataset, scene_idx):
    indexes = dataset.get_scene_indices(scene_idx)
    return create_animate_for_indexes(dataset, indexes)

# Prepare all rasterizer and EgoDataset for each rasterizer
rasterizer_dict = {}
dataset_dict = {}

rasterizer_type_list = ["py_satellite", "satellite_debug", "py_semantic", "semantic_debug", "box_debug", "stub_debug"]

for i, key in enumerate(rasterizer_type_list):
    # print("key", key)
    cfg["raster_params"]["map_type"] = key
    rasterizer_dict[key] = build_rasterizer(cfg, dm)
    dataset_dict[key] = EgoDataset(cfg, zarr_dataset, rasterizer_dict[key])


# default lane color is "light yellow" (255, 217, 82).
# green, yellow, red color on lane is to show trafic light condition.
# orange box represents crosswalk

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, key in enumerate(["stub_debug", "satellite_debug", "semantic_debug", "box_debug", "py_satellite", "py_semantic"]):
    visualize_rgb_image(dataset_dict[key], index=0, title=f"{key}: {type(rasterizer_dict[key]).__name__}", ax=axes[i])
fig.show()

# Scenes animations
# That will work just fine in the notebook
dataset = dataset_dict["py_semantic"]
plt.rcParams['animation.embed_limit'] = 4**128
SceneIndex = 10
print("scene_idx", SceneIndex)
anim = create_animate_for_scene(dataset, SceneIndex)
display(HTML(anim.to_jshtml()))


