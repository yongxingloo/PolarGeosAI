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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split

from .func_ml import *
from .impl import *
from .MultiInputModel import *

# for single channel testing


def train_test(
    data_file,
    model_parameters,
):

    data = np.load(data_file)
    images = data["images"]
    numerical_data = data["numerical_data"].T

    batch_size = model_parameters["batch_size"]
    image_height = model_parameters["image_height"]
    image_width = model_parameters["image_width"]
    num_numerical_inputs = model_parameters["num_numerical_inputs"]
    features_cnn = model_parameters["features_cnn"]
    features_numerical = model_parameters["features_numerical"]
    kernel_size = model_parameters["kernel_size"]
    in_channels = model_parameters["image_channels"]
    activation_cnn = model_parameters["activation_cnn"]
    activation_numerical = model_parameters["activation_numerical"]
    activation_final = model_parameters["activation_final"]
    stride = model_parameters["stride"]
    lr = model_parameters["learning_rate"]
    weight_decay = model_parameters["weight_decay"]
    criterion = model_parameters["criterion"]
    optimizer_choice = model_parameters["optimizer_choice"]
    dropout_rate = model_parameters["dropout_rate"]
    normalization = model_parameters["normalization"]
    num_epochs = 800

    # Introduce normalization skip for target
    wind_speeds = numerical_data[:, -1]
    # Normalize data
    if normalization:
        _images, numerical_data, image_means, image_stds, num_means, num_stds = (
            normalize_data(images, numerical_data)
        )
        print(
            "Data normalized !"
        )  # only numerical data. Images are not normalized because decreases the performance (cause probably binomial dist.)

    # normalize data skip
    numerical_data[:, -1] = wind_speeds

    experiment_name = data_file.split("/")[-1].split(".")[0]
    path_folder = create_folder(experiment_name)

    """Network Parameters"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("image height:", image_height)
    print("image width:", image_width)
    print("in channels:", in_channels)
    print("num numerical inputs:", num_numerical_inputs)
    print("criterion:", criterion)
    print("optimizer choice:", optimizer_choice)
    print("batch size:", batch_size)
    print("lr:", lr)
    print("weight decay (regularization):", weight_decay)
    print("features numerical:", features_numerical)
    print("features cnn:", features_cnn)
    print("kernel size:", kernel_size)
    print("stride:", stride)
    print("activation numerical:", activation_numerical)
    print("activation cnn:", activation_cnn)
    print("activation final:", activation_final)

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
        dropout_rate,
    ).to(device)

    # Early stopping parameters

    patience_epochs = 50
    patience_loss = 0.001

    # Create directory for all saving

    print("running on ", device)

    # split the data

    images_train, images_test, numerical_data_train, numerical_data_test = (
        train_test_split(images, numerical_data, test_size=0.2, random_state=42)
    )

    # Bring the data into the expected format

    dataset = MultiInputDataset(
        images_train,
        numerical_data_train,
        transform=None,
    )

    print("Data loaded !")

    # Train / Test split

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    print(train_size, " = train size")
    print(val_size, " = val size")

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True, drop_last=False)

    # Train the model
    print("Start of training !")

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
        numerical_data_test,
        transform=None,
    )
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    wind_speeds_test = numerical_data_test[:, -1]

    test_output, test_loss = test_model(model, test_loader, criterion, device)

    error_plot(test_output, wind_speeds_test, path_folder)

    return wind_speeds_test, test_output, test_loss


def pretrained_test(data_file, model_parameters, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(data_file)
    images = data["images"]
    numerical_data = data["numerical_data"].T

    batch_size = model_parameters["batch_size"]
    image_height = model_parameters["image_height"]
    image_width = model_parameters["image_width"]
    num_numerical_inputs = model_parameters["num_numerical_inputs"]
    features_cnn = model_parameters["features_cnn"]
    features_numerical = model_parameters["features_numerical"]
    kernel_size = model_parameters["kernel_size"]
    in_channels = model_parameters["image_channels"]
    activation_cnn = model_parameters["activation_cnn"]
    activation_numerical = model_parameters["activation_numerical"]
    activation_final = model_parameters["activation_final"]
    stride = model_parameters["stride"]
    criterion = model_parameters["criterion"]
    dropout_rate = model_parameters["dropout_rate"]

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
        dropout_rate,
    ).to(device)

    model.load_state_dict(torch.load(model_path))
    criterion = nn.MSELoss()

    wind_speeds = numerical_data[:, -1]

    # Normalize data
    _images, numerical_data, image_means, image_stds, num_means, num_stds = (
        normalize_data(images, numerical_data)
    )
    print(
        "Data normalized !"
    )  # only numerical data. Images are not normalized because decreases the performance (cause probably binomial dist.)

    # normalize data skip
    numerical_data[:, -1] = wind_speeds

    test_dataset = MultiInputDataset(
        images,
        numerical_data,
        transform=None,
    )
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    test_output, test_loss = test_model(model, test_loader, criterion, device)

    error_plot(test_output, wind_speeds)

    return wind_speeds, test_output, test_loss
