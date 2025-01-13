"""
@author: Yongxing Loo
"""

import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split

from .func_ml import *
from .impl import *
from .MultiInputModel import *


## Data input 




def train_multi_input_model(data_file):

    data = np.load(data_file)
    images = data["images"]
    # Create dummy channel
    C02 = np.zeros(images.shape)
    images = np.concatenate((images, C02), axis=1)
    numerical_data = data["numerical_data"]
    saa = numerical_data[0]
    sza = numerical_data[1]
    wind_speeds = numerical_data[2]



    experiment_name = data_file.split("/")[-1].split(".")[0]
    path_folder = create_folder(experiment_name)


        
    """Network Parameters"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_height = 30
    image_width = 30
    in_channels = 2  # Number of channels in the input image (C01 and C03)
    num_numerical_inputs = 2 # Number of numerical inputs (lat, lon, hour(sin), hour(cos), day(sin), day(cos))
    num_epochs = 800


    # IN GRIDSEARCH

    criterion = nn.MSELoss()  # "MSELoss", "L1Loss"
    optimizer_choice = "Adam"  # "Adam", "SGD", "RMSprop"

    batch_size = 512
    lr = 0.001
    weight_decay = 0.001 # L2 regularization

    features_numerical = [4, 8, 16,32]
    features_cnn = [32, 64, 128]

    kernel_size = 3
    stride = 1

    activation_numerical = nn.ReLU()
    activation_cnn = nn.ReLU()
    activation_final = nn.Identity()


    model = MultiInputModel(
        image_height,
        image_width,
        num_numerical_inputs,
        features_cnn,
        features_numerical,
        kernel_size,
        in_channels,
        activation_cnn,
        activation_numerical,
        activation_final,
        stride,
    ).to(device)

    # Early stopping parameters

    patience_epochs = 50
    patience_loss = 0.001

    # Create directory for all saving

    print('running on ', device)
    
    # split the data
    
    images_train, images_test, saa_train, saa_test,sza_train, sza_test,wind_speeds_train,wind_speeds_test = train_test_split(
        images, saa, sza,wind_speeds, test_size=0.2, random_state=42
    )

    
    # Bring the data into the expected format

    dataset = MultiInputDataset(
        images_train, saa_train, sza_train, wind_speeds_train, transform=None, 
        
    )
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)


    for images, numerical_input, target in dataloader:
        print(images.shape)
        print(numerical_input.shape)
        print(target.shape)
        print("example target:", target)
        print("example numerical input:", numerical_input)
        break

    print("Data loaded !")

    # Train / Test split
    print("Start of training !")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, drop_last=False)

    # Train the model


    best_val_outputs, best_val_labels, model, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs,
        lr,
        weight_decay,
        criterion,
        device,
        optimizer_choice,
        patience_epochs,
        patience_loss,
        path_folder,
    )

    print("Training done !")

    # Ploting and saving the results

    plot_save_loss(
        best_val_outputs,
        best_val_labels,
        train_losses,
        val_losses,
        path_folder,
        saving=False,
    )

# Test the model



    test_dataset = MultiInputDataset(
        images_test,
        saa_test,
        sza_test,
        wind_speeds_test,
        transform=None,
    )
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    test_output, test_loss = test_model(model, test_loader, criterion, device)

    error_plot(test_output, wind_speeds_test, path_folder)

    score_test = MSE(test_output, wind_speeds_test)

    return wind_speeds_test