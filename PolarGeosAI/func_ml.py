import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import torch
import torch.utils.data
from pysolar.solar import *


def create_folder(experiment_name):
    date_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    path_folder = f"../results_folder/{date_time}_{experiment_name}"

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
        print(f"Folder created at {path_folder}")

    return path_folder


def lat_lon_to_xyz(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    xcoord = np.cos(lat_rad) * np.cos(lon_rad)
    ycoord = np.cos(lat_rad) * np.sin(lon_rad)
    zcoord = np.sin(lat_rad)
    return xcoord, ycoord, zcoord


def solar_zenith_angle(lat, lon, datetime):
    date = pd.to_datetime(datetime).tz_localize(pytz.utc)
    sza = np.zeros(len(date))
    for i in range(len(date)):
        sza[i] = 90 - get_altitude(lat[i], lon[i], date[i])

    return sza


def solar_azimuth_angle(lat, lon, datetime):
    date = pd.to_datetime(datetime).tz_localize(pytz.utc)
    saa = np.zeros(len(date))
    for i in range(len(date)):
        saa[i] = get_azimuth(lat[i], lon[i], date[i])

    return saa


def normalization_mean_std(images_dual_channel, target):
    # Create a mask to exclude zero values
    mask = images_dual_channel != 0

    # Compute mean and standard deviation for non-zero values only
    image_mean = (images_dual_channel * mask).sum(
        axis=(0, 2, 3), keepdims=True
    ) / mask.sum(axis=(0, 2, 3), keepdims=True)
    image_std = np.sqrt(
        ((images_dual_channel * mask - image_mean) ** 2).sum(
            axis=(0, 2, 3), keepdims=True
        )
        / mask.sum(axis=(0, 2, 3), keepdims=True)
    )

    # Compute mean and std for target normally (if it doesn't require masking)
    target_mean, target_std = target.mean(), target.std()

    # Normalize images
    images_dual_channel = (images_dual_channel - image_mean) / image_std

    # Normalize target
    target = (target - target_mean) / target_std

    return images_dual_channel, target, image_mean, image_std, target_mean, target_std


def normalization_mean_std_test(
    images_dual_channel_test,
    target_test,
    image_mean,
    image_std,
    target_mean,
    target_std,
):

    # Normalize images
    images_dual_channel_test = (images_dual_channel_test - image_mean) / image_std

    # Normalize target
    target_test = (target_test - target_mean) / target_std

    return images_dual_channel_test, target_test


def normalization_solar_angles(saa, sza):

    saa_mean = saa.mean()
    saa_std = saa.std()

    sza_mean = sza.mean()
    sza_std = sza.std()

    saa = (saa - saa_mean) / saa_std
    sza = (sza - sza_mean) / sza_std

    return saa, sza, saa_mean, saa_std, sza_mean, sza_std


def normalization_solar_angles_test(
    saa_test, sza_test, saa_mean, saa_std, sza_mean, sza_std
):

    saa_test = (saa_test - saa_mean) / saa_std
    sza_test = (sza_test - sza_mean) / sza_std

    return saa_test, sza_test


def normalisation_minmax(images_dual_channel, target):

    image_min = images_dual_channel.min(axis=(0, 2, 3), keepdims=True)
    image_max = images_dual_channel.max(axis=(0, 2, 3), keepdims=True)

    target_min, target_max = target.min(), target.max()

    # Normalize images
    images_dual_channel = (images_dual_channel - image_min) / (image_max - image_min)

    # Normalize target
    target = (target - target_min) / (target_max - target_min)

    return images_dual_channel, target, image_min, image_max, target_min, target_max


def normalisation_minmax_test(
    images_dual_channel, target, image_min, image_max, target_min, target_max
):
    # Normalize images
    images_dual_channel = (images_dual_channel - image_min) / (image_max - image_min)

    # Normalize target
    target = (target - target_min) / (target_max - target_min)

    return images_dual_channel, target


def normalization_solar_angles(saa, sza):
    saa_min = saa.min()
    saa_max = saa.max()
    sza_min = sza.min()
    sza_max = sza.max()

    saa = (saa - saa_min) / (saa_max - saa_min)
    sza = (sza - sza_min) / (sza_max - sza_min)

    return saa, sza, saa_min, saa_max, sza_min, sza_max


class DateTimeCyclicEncoder:
    def __call__(self, datetime_value):
        # Ensure datetime_value is of dtype datetime64 for compatibility
        datetime_value = np.datetime64(
            datetime_value, "ns"
        )  # Convert to nanosecond precision if needed

        # Calculate day of year
        year_start = datetime_value.astype("datetime64[Y]").astype("datetime64[D]")
        day_of_year = (datetime_value.astype("datetime64[D]") - year_start).astype(int)

        # Calculate hour of day
        day_start = datetime_value.astype("datetime64[D]")
        hour_of_day = ((datetime_value - day_start) / np.timedelta64(1, "h")).astype(
            int
        )

        # Cyclic encoding with sine and cosine transformations
        day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365)
        day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365)
        hour_of_day_sin = np.sin(2 * np.pi * hour_of_day / 24)
        hour_of_day_cos = np.cos(2 * np.pi * hour_of_day / 24)

        # Return as a tensor for compatibility with PyTorch
        return torch.tensor(
            [day_of_year_sin, day_of_year_cos, hour_of_day_sin, hour_of_day_cos],
            dtype=torch.float32,
        )


time_transform = DateTimeCyclicEncoder()


class MultiInputDataset(torch.utils.data.Dataset):
    def __init__(
        self, images, saa, sza, targets=None, transform=None
    ):
        self.images = images
        self.saa = saa
        self.sza = sza
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        numerical_input = torch.tensor(
            [self.saa[idx], self.sza[idx]], dtype=torch.float32
        )

        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return image, numerical_input, target
        else:
            return image, numerical_input


def visualize_kernels(layer_weights, layer_name):
    num_kernels = layer_weights.size(0)
    num_channels = layer_weights.size(1)

    # Use symmetric vmin and vmax to preserve sign information
    abs_max = max(abs(layer_weights.min().item()), abs(layer_weights.max().item()))
    vmin, vmax = -abs_max, abs_max

    # Define grid size for plotting
    rows = int(np.sqrt(num_kernels))
    cols = int(np.ceil(num_kernels / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i in range(num_kernels):
        ax = axes[i]

        # Select the first channel for visualization
        kernel = layer_weights[i, 0].detach().cpu().numpy()

        # Plot the kernel using a diverging colormap
        im = ax.imshow(kernel, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.axis("off")
        ax.set_title(f"Kernel {i + 1}")
        fig.colorbar(im, ax=ax)

    # Hide any unused subplots
    for j in range(num_kernels, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle(f"Kernels of {layer_name}")
    plt.savefig(f"../interpretibility/kernels_{layer_name}.png")
    plt.show()


def visualize_numerical_layers(layer_weights, layer_name):
    # Get the weights and biases
    weights = layer_weights.weight.detach().cpu().numpy()
    biases = layer_weights.bias.detach().cpu().numpy()

    # Define the figure size
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the weights as a heatmap
    cax = ax.matshow(weights, cmap="coolwarm", aspect="auto")

    # Add colorbar
    fig.colorbar(cax)

    # Set axis labels
    ax.set_xlabel("Input Features")
    ax.set_ylabel("Output Neurons")
    ax.set_title(f"Weights of {layer_name}")

    # Add grid lines to separate output neurons
    for i in range(weights.shape[0]):
        ax.axhline(i - 0.5, color="black", linewidth=0.5)

    # Show the plot
    plt.savefig(f"../interpretibility/weights_{layer_name}.png")
    plt.show()

    # Plot the biases
    plt.figure(figsize=(10, 2))
    plt.bar(range(len(biases)), biases, color="blue")
    plt.xlabel("Output Neurons")
    plt.ylabel("Bias Value")
    plt.title(f"Biases of {layer_name}")
    plt.savefig(f"../interpretibility/biases_{layer_name}.png")
    plt.show()


def error_plot(best_val_outputs, best_val_labels, path_folder):
    error = best_val_outputs - best_val_labels
    x_error = np.arange(0, len(error))

    plt.figure(figsize=(10, 5))
    plt.title("Error in Wind Speed Prediction")
    plt.plot(x_error, error)
    plt.ylabel("Error in wind speed prediction in m/s")
    plt.xlabel("Sample")
    plt.grid()

    mean = np.mean(error)
    std = np.std(error)

    plt.axhline(y=mean, color="r", linestyle=":", label="Mean Error")
    plt.axhline(y=mean + std, color="g", linestyle="--", label="Mean Error + Std")
    plt.axhline(y=mean - std, color="g", linestyle="--", label="Mean Error - Std")
    plt.legend()
    plt.savefig(os.path.join(path_folder, "error_plot.png"))

    plt.figure()
    plt.plot(best_val_labels, best_val_outputs, "o")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("True Labels")
    plt.ylabel("Model Output")
    plt.title("Model Output vs True Labels")
    plt.xticks(np.arange(0, 30, 5))
    plt.yticks(np.arange(0, 30, 5))
    plt.plot(
        [min(best_val_labels), max(best_val_labels)],
        [min(best_val_labels), max(best_val_labels)],
        "r--",
    )  # y = x reference line
    plt.savefig(os.path.join(path_folder, "scatter_plot.png"))
    plt.show()


def plot_save_loss(
    best_val_outputs, best_val_labels, train_losses, val_losses, path_folder,saving=False
):
    # After training, save only the best validation outputs and labels
    if saving:
        np.save(os.path.join(path_folder, "best_validation_outputs.npy"), best_val_outputs)
        np.save(os.path.join(path_folder, "best_validation_labels.npy"), best_val_labels)

    num_epochs = len(train_losses)
    # Plotting the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    text_str = f"num_epochs = {num_epochs}, train loss = {train_losses[-1]:.2f}, validation loss = {val_losses[-1]:.2f}"
    plt.text(
    0.05, 0.05, text_str,
    ha="left", va="bottom", 
        transform=plt.gca().transAxes  # Ensures the coordinates are relative to the axes (0 to 1 range)
    )
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(path_folder, "loss_plot.png"))


def MSE(y_true, y_pred):
    print(np.mean((y_true - y_pred) ** 2), "= MSE after run")
    return np.mean((y_true - y_pred) ** 2)
