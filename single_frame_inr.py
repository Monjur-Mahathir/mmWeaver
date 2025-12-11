import os 
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 

from src.utils import *
from src.model import *
from src.dataloader import *
from src.metrics import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)


def main():
    frame_path = "Hupr/Processed/single_1/hori/000000001.npy"
    num_epochs = 1024
    bs = 16384

    radar_data = SingleFrameDataset(frame_path=frame_path)
    indexed_dataloader = DataLoader(radar_data, batch_size=bs, shuffle=True)

    model = INRNet(
        positional_embedding='ffm', 
        in_features=4, 
        out_features=2, 
        hidden_layers=4, 
        hidden_features=256, 
        nonlinearity='relu', 
        map_size=2048, 
        map_scale=256
    )

    optim = torch.optim.Adam(lr=1e-3, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=32, gamma=0.99)

    model = model.to(device)
    print("Starting Training ...")

    best_loss = 999999
    for step in range(num_epochs):
        model.train()
        loss_val = 0
        start_time = time.time()
        #"""
        # Coordinate-wise training
        for i, data in enumerate(indexed_dataloader):
            coordinates, values = data 
            
            coordinates = coordinates.to(device)
            values = values.to(device)
            
            optim.zero_grad()

            values_pred = model(coordinates)
            
            loss_real = ((values_pred[..., 0] - values[..., 0]) ** 2)
            loss_imag = ((values_pred[..., 1] - values[..., 1]) ** 2)
            loss_val = loss_real.sum().item() + loss_imag.sum().item() + loss_val

            loss = 1000 * (loss_real.mean() + loss_imag.mean())

            loss.backward()
            optim.step()
        loss_val = loss_val / (bs * (i+1))
        
        if step % 64 == 0:
            print(f"Step: {step} Coordinate-wise loss: {loss_val}")
            print(f"Training time for each epoch: {(time.time() - start_time):.2f} seconds")

        if loss_val < best_loss:
            best_loss = loss_val
            model_path = 'weights/single_frame_inr'
            torch.save(model.state_dict(), model_path)
        
        scheduler.step()


if __name__ == "__main__":
    main()