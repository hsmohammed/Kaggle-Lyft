#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 20:24:16 2020

@author: Hossameldin Mohammed & Mohamed Kamel
"""
import matplotlib.pyplot as plt
from matplotlib import animation, rc
rc('animation', html='jshtml')
from IPython.display import HTML
#from IPython.display import display, clear_output
import PIL
import zarr
from pathlib import Path
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
#import scipy as sp
import itertools as it
import seaborn as sns
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
#from tqdm import tqdm
#from collections import Counter
from l5kit.data import PERCEPTION_LABELS
#from prettytable import PrettyTable

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


cfg2 = omegaconf.DictConfig(cfg2)


frames = zarr_dataset.frames
agents = zarr_dataset.agents
scenes = zarr_dataset.scenes
# tl_faces = zarr_dataset.tl_faces


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


# Agents EDA

agent = agents[0]
PERCEPTION_LABELS = [
    "NOT_SET",
    "UNKNOWN",
    "DONTCARE",
    "CAR",
    "VAN",
    "TRAM",
    "BUS",
    "TRUCK",
    "EMERGENCY_VEHICLE",
    "OTHER_VEHICLE",
    "BICYCLE",
    "MOTORCYCLE",
    "CYCLIST",
    "MOTORCYCLIST",
    "PEDESTRIAN",
    "ANIMAL",
    "DONTCARE",
]
DATA_ROOT = Path("C:/Users/Omar/Documents/GitHub/Kaggle-Lyft/data")
#A robust and fast interface to load l5kit data into  Pandas dataframes
class BaseParser:

    field = "scenes"
    dtypes = {}

    def __init__(self, start=0, end=None, chunk_size=1000, max_chunks=1000, root=DATA_ROOT,
                 zarr_path="scenes/sample.zarr"):

        self.start = start
        self.end = end
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks

        self.root = Path(root)
        assert self.root.exists(), "There is nothing at {}!".format(self.root)
        self.zarr_path = Path(zarr_path)

    def parse(self):
        raise NotImplementedError

    def to_pandas(self, start=0, end=None, chunk_size=None, max_chunks=None):
        start = start or self.start
        end = end or self.end
        chunk_size = chunk_size or self.chunk_size
        max_chunks = max_chunks or self.max_chunks

        if not chunk_size or not max_chunks:  # One shot load, suitable for small zarr files
            df = zarr.load(self.root.joinpath(self.zarr_path).as_posix()).get(self.field)
            df = df[start:end]
            df = map(self.parse, df)
        else:  # Chunked load, suitable for large zarr files
            df = []
            with zarr.open(self.root.joinpath(self.zarr_path).as_posix(), "r") as zf:
                end = start + max_chunks * chunk_size if end is None else min(end, start + max_chunks * chunk_size)
                for i_start in range(start, end, chunk_size):
                    items = zf[self.field][i_start: min(i_start + chunk_size, end)]
                    items = map(self.parse, items)
                    df.append(items)
            df = it.chain(*df)

        df = pd.DataFrame.from_records(df)
        for col, col_dtype in self.dtypes.items():
            df[col] = df[col].astype(col_dtype, copy=False)
        return df

class AgentParser(BaseParser):
    field = "agents"

    @staticmethod
    def parse(agent):
        frame_dict = {
            'centroid_x': agent[0][0],
            'centroid_y': agent[0][1],
            'extent_x': agent[1][0],
            'extent_y': agent[1][1],
            'extent_z': agent[1][2],
            'yaw': agent[2],
            "velocity_x": agent[3][0],
            "velocity_y": agent[3][1],
            "track_id": agent[4],
        }
        for p_label, p in zip(PERCEPTION_LABELS, agent[5]):
            frame_dict["{}".format(p_label)] = p
        return frame_dict

    def to_pandas(self, start=0, end=None, chunk_size=None, max_chunks=None, frame=None):
        if frame is not None:
            start = int(frame.agent_index_interval_start)
            end = int(frame.agent_index_interval_end)

        df = super().to_pandas(start=start, end=end, chunk_size=chunk_size, max_chunks=max_chunks)
        return df

ap = AgentParser()
agents_df = ap.to_pandas(frame=None)
agents_df.head()
agents_df.columns
# Agents EDA
agents_df.describe()
agents_df.shape
agents_df.info()
agents_labels = [agents_df.CAR.sum(), agents_df.PEDESTRIAN.sum(), agents_df.CYCLIST.sum()]
colormap = plt.cm.magma
corr_matrix = ["centroid_x", "centroid_y", "extent_x", "extent_y", "extent_z", "yaw", 'velocity_x', 'velocity_y', "CAR","PEDESTRIAN","CYCLIST"  ]
plt.figure(figsize=(20,20));
plt.title('Pearson correlation of features', y=1.0, size=14);
sns.heatmap(agents_df[corr_matrix].corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
sns.scatterplot(agents_df["centroid_x"],agents_df["centroid_y"])



agents_CPC_df = agents_df.loc[((agents_df.CAR>= 0.5)|(agents_df.PEDESTRIAN>= 0.5)|(agents_df.CYCLIST>= 0.5))]
agents_CPC_df.info()

agents_CPC_df.centroid_x.idxmax()


colormap = plt.cm.magma
corr_matrix = ["centroid_x", "centroid_y", "extent_x", "extent_y", "extent_z", "yaw", 'velocity_x', 'velocity_y', "CAR","PEDESTRIAN","CYCLIST"  ]
plt.figure(figsize=(20,20));
plt.title('Pearson correlation of features', y=1.0, size=14);
sns.heatmap(agents_CPC_df[corr_matrix].corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


agents_CAR_df = agents_df.loc[((agents_df.CAR>= 0.5))]
agents_CAR_df.info()

colormap = plt.cm.magma
corr_matrix = ["centroid_x", "centroid_y", "extent_x", "extent_y", "extent_z", "yaw", 'velocity_x', 'velocity_y']
plt.figure(figsize=(20,20));
plt.title('Pearson correlation of features', y=1.0, size=14);
sns.heatmap(agents_CAR_df[corr_matrix].corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)



import numpy as np
a = np.array([1,2,3,4])
a.shape
import time
a = np.random.rand(1000000)
b = np.random.rand(1000000)
tic = time.time()
c = np.dot(a,b)
toc = time.time()
print(f'vectorized ver.: {str(1000*(toc-tic))} ms')



c = 0
tic = time.time()
for i in range(1000000):
    c+=a[i]*b[i]
toc = time.time()
print(f'vectorized ver.: {str(1000*(toc-tic))} ms')

a.reshape(4,1)