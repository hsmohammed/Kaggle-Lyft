#import os

#os.system('pip install --target=/kaggle/working pymap3d==2.1.0 --upgrade')
#os.system('pip install --target=/kaggle/working protobuf==3.12.2 --upgrade')
#os.system('pip install --target=/kaggle/working transforms3d --upgrade')
#os.system('pip install --target=/kaggle/working zarr --upgrade')
#os.system('pip install --target=/kaggle/working ptable --upgrade')

#os.system('pip install --no-dependencies --target=/kaggle/working l5kit --upgrade')

from typing import Dict
import torchvision.models as models
from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101
from tqdm import tqdm
from torchvision import models
import l5kit
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path

import matplotlib.pyplot as plt

import os
import random
import time

import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import torch.optim as optim

l5kit.__version__



# --- Lyft configs ---
cfg = {
    'format_version': 4,
    'data_path': "data",
    'model_params': {
        'model_architecture': 'resnet34',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'model_name': "model_resnet34_output",
        'lr': 1e-3,
        'weight_path': False,# "data/model_multi_update_lyft_public.pth"
        'train': True,
        'predict': True
    },

    'raster_params': {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },

    'train_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 0
    },

    'val_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 0
    },

    'full_data_loader': {
        'key': 'scenes.full/train_full.zarr',
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 0
    },

    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 16,
        'shuffle': False,
        'num_workers': 0
    },

    'train_params': {
        'max_num_steps': 4000, # number of epochs # 65,536 *16 (Batch size) 1M Sample
        'checkpoint_every_n_steps': 40,
    }
}


"""
Couple of things to note:

model_architecture: you can put 'resnet18', 'resnet34' or 'resnet50'. For the pretrained model we use resnet18 so we need to use 'resnet18' in the config.
weight_path: path to the pretrained model. If you don't have a pretrained model and want to train from scratch, put weight_path = False.
model_name: the name of the model that will be saved as output, this is only when train= True.
train: True if you want to continue to train the model. Unfortunately due to Kaggle memory constraint if train=True then you should put predict = False.
predict: True if you want to predict and submit to Kaggle. Unfortunately due to Kaggle memory constraint if you want to predict then you need to put train = False.
lr: learning rate of the model, feel free to change as you see fit. In the future I also plan to implement learning rate decay.
raster_size: specify the size of the image, the default is [224,224]. Increase raster_size can improve the score. However the training time will be significantly longer.
batch_size: number of inputs for one forward pass, again one of the parameters to tune.
max_num_steps: the number of iterations to train, i.e. number of epochs.
checkpoint_every_n_steps: the model will be saved at every n steps, again change this number as to how you want to keep track of the model.
"""


# set env variable for data
DIR_INPUT = cfg["data_path"]
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)

# ===== INIT TRAIN DATASET============================================================
train_cfg = cfg["train_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open(cached=False)
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],num_workers=train_cfg["num_workers"])

print("==================================TRAIN DATA==================================")
print(train_dataset)



# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py
import numpy as np

import torch
from torch import Tensor


def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,)), atol=1e-06), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences+ 1e-10) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error)


def pytorch_neg_multi_log_likelihood_single(
    gt: Tensor, pred: Tensor, avails: Tensor
) -> Tensor:
    """

    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return pytorch_neg_multi_log_likelihood_batch(gt, pred.unsqueeze(1), confidences, avails)




class LyftMultiModel(nn.Module):

    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels




        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len

        self.image_pretrained = models.resnet18(pretrained=True)

        #for name, param in self.image_pretrained.named_parameters():
        #    if ("bn" not in name):
        #        param.requires_grad = False

        #for param in self.image_pretrained.parameters():
        #   param.requires_grad = False
        self.bn2d = nn.BatchNorm2d(num_in_channels, eps=1e-05, momentum=None)
        self.image_pretrained.conv1 = nn.Conv2d(
            num_in_channels,
            self.image_pretrained.conv1.out_channels,
            kernel_size=self.image_pretrained.conv1.kernel_size,
            stride=self.image_pretrained.conv1.stride,
            padding=self.image_pretrained.conv1.padding,
            bias=True,
        )
        backbone_out_features = self.image_pretrained.fc.in_features # depends on the Arch used in the first part

        self.image_pretrained.fc = nn.Sequential(nn.BatchNorm1d(backbone_out_features, eps=1e-05, momentum=None),nn.Dropout(0.2),
                                                 nn.Linear(backbone_out_features, 1024),nn.ReLU()
                                                 ,nn.Linear(1024,2500),nn.ReLU())

        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.lstm1 = nn.LSTM(input_size=100, hidden_size=100, batch_first=True, num_layers=5)
        self.conv2d = nn.Sequential(nn.Conv2d(num_in_channels,64, kernel_size= 7, stride=2,padding=2,bias=True),
                                    nn.BatchNorm2d(64, eps=1e-05, momentum=None),nn.ReLU(),
                                    nn.Conv2d(64,25, kernel_size= 3, stride=2,padding=2,bias=True),
                                    nn.BatchNorm2d(25, eps=1e-05, momentum=None),nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False))

        self.lstm2 = nn.LSTM(input_size=841, hidden_size=512, batch_first=True, num_layers=3)
        self.lstm3 = nn.LSTM(input_size=612, hidden_size=256, batch_first=True, num_layers=1)

        # You can add more layers here.
        self.head = nn.Sequential(nn.BatchNorm1d(6400, eps=1e-05, momentum=None),nn.ReLU(),
                                  nn.Linear(in_features=6400, out_features=4096),nn.ReLU(),
                                  nn.Linear(in_features=4096, out_features=2048),nn.ReLU(),
                                  nn.Linear(in_features=2048, out_features=1024),nn.ReLU()
                                  )


        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(1024, out_features=self.num_preds + num_modes)

    def forward(self, x):
        x1 = self.image_pretrained(x)
        x1, _ = self.lstm1(x1.view(len(x1), 25, -1))
        x2 = self.conv2d(x)
        x2, _ = self.lstm2(x2.view(len(x2), 25, -1))
        x = torch.cat((x1,x2), dim=2)
        x, _ = self.lstm3(x.view(len(x), 25, -1))
        x = torch.reshape(x, (len(x), 6400))
        x = self.head(x)
        x = self.logit(x)
        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = x.shape # batch size
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences


def forward(data, model, device, criterion = pytorch_neg_multi_log_likelihood_batch):
    inputs = data["image"].to(device) #X
    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device) #Y
    # Forward pass
    preds, confidences = model(inputs)
    loss = criterion(targets, preds, confidences, target_availabilities)
    return loss, preds, confidences

# ==== INIT MODEL=================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LyftMultiModel(cfg)

simplenet_state_dict = torch.load("Ong_Model_Two_way_LSTM.pth")
model.load_state_dict(simplenet_state_dict)

opt_state_dict = torch.load("Ong_optimizer_Two_way_LSTM.pth")# load optimizer





#load weight if there is a pretrained model
# ==== INIT MODEL=================
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = LyftMultiModel(cfg)
#load weight if there is a pretrained model
#weight_path = cfg["model_params"]["weight_path"]
#if weight_path:
#    model.load_state_dict(torch.load(weight_path))

model.to(device)
#optimizer = optim.SGD(model.parameters(), lr=cfg["model_params"]["lr"], momentum=0.9)

#optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])
lr = cfg["model_params"]["lr"] = 0.001
optimizer = optim.Adam(params=model.parameters(), betas=(0.9, 0.9999), lr=cfg["model_params"]["lr"], amsgrad=True)
optimizer.load_state_dict(opt_state_dict)

print(f'device {device}')
#weight_path = cfg["model_params"]["weight_path"]
#if weight_path:
#    model.load_state_dict(torch.load(weight_path))

#model.to(device)
#optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])
#print(f'device {device}')




pytorch_total_params = sum(p.numel() for p in model.parameters())  # all parmeters
pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) #trainable
print(f'total parameters={pytorch_total_params}, trainable ={pytorch_total_trainable_params}')



print(model)
tr_it = iter(train_dataloader)
__z =60000



it = cfg["train_params"]["max_num_steps"] = 4001
bss = train_cfg["batch_size"]
cfg["model_params"]["model_name"] = "Two_way_LSTM"
__z += cfg["train_params"]["max_num_steps"]

# +(it*bss)
#train_Subset = Subset(train_dataset,range(500000, 616000))
#train_dataloader = DataLoader(train_Subset, shuffle=True, batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"])





# ==== TRAINING LOOP =========================================================

progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
num_iter = cfg["train_params"]["max_num_steps"]
losses_train = []
iterations = []
metrics = []
times = []
model_name = cfg["model_params"]["model_name"]
start = time.time()
for i in progress_bar:
    try:
        data = next(tr_it)
    except StopIteration:
        #tr_it = iter(train_dataloader)
        data = next(tr_it)
    model.train()
    torch.set_grad_enabled(True)

    loss, _, _ = forward(data, model, device)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses_train.append(loss.item())

    progress_bar.set_description(f"loss: {loss.item() } loss(avg): {np.mean(losses_train)}")
    if i % (cfg['train_params']['checkpoint_every_n_steps']) == 0:
        if i % (cfg['train_params']['checkpoint_every_n_steps'] * 25) == 0:
            torch.save(model.state_dict(), f'{model_name}_{i}_{__z}.pth')
            torch.save(optimizer.state_dict(), f'optimizer_{model_name}_{i}_{__z}.pth')

        torch.save(model.state_dict(), f'Ong_Model_{model_name}.pth')
        torch.save(optimizer.state_dict(), f'Ong_optimizer_{model_name}.pth')
        iterations.append(i)
        metrics.append(np.mean(losses_train))
        times.append((time.time() - start) / 60)
        losses_train = []
        results = pd.DataFrame({'iterations': iterations, 'metrics (avg)': metrics, 'elapsed_time (mins)': times})
        results.to_csv(f"train_metrics_{model_name}_{num_iter}_{__z}.csv", index=False)


results = pd.DataFrame({'iterations': iterations, 'metrics (avg)': metrics, 'elapsed_time (mins)': times})
#results.to_csv(f"train_metrics_{model_name}_{num_iter}_{z}_{lr}_.csv", index=False)
print(f"Total training time is {(time.time() - start) / 60} mins")
print(results)


#====== INIT TEST DATASET=============================================================
test_cfg = cfg["test_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open(cached=False)
test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]
test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
test_dataloader = DataLoader(test_dataset,shuffle=test_cfg["shuffle"],batch_size=test_cfg["batch_size"],
                             num_workers=test_cfg["num_workers"])
print("==================================TEST DATA==================================")
print(test_dataset)


if cfg["model_params"]["predict"]:

    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []
    confidences_list = []
    agent_ids = []

    progress_bar = tqdm(test_dataloader)

    for data in progress_bar:

        _, preds, confidences = forward(data, model, device)

        # fix for the new environment
        preds = preds.cpu().numpy()
        world_from_agents = data["world_from_agent"].numpy()
        centroids = data["centroid"].numpy()
        coords_offset = []

        # convert into world coordinates and compute offsets
        for idx in range(len(preds)):
            for mode in range(3):
                preds[idx, mode, :, :] = transform_points(preds[idx, mode, :, :], world_from_agents[idx]) - centroids[
                                                                                                                idx][:2]

        future_coords_offsets_pd.append(preds.copy())
        confidences_list.append(confidences.cpu().numpy().copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())


#create submission to submit to Kaggle
pred_path = 'submission.csv'
write_pred_csv(pred_path,
           timestamps=np.concatenate(timestamps),
           track_ids=np.concatenate(agent_ids),
           coords=np.concatenate(future_coords_offsets_pd),
           confs = np.concatenate(confidences_list)
          )
