import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn, optim


def early_stopping(valid_losses, patience_epochs, patience_loss):  # From @Jing
    if len(valid_losses) < patience_epochs:
        return False
    recent_losses = valid_losses[-patience_epochs:]

    if all(x >= recent_losses[0] for x in recent_losses):
        return True

    if max(recent_losses) - min(recent_losses) < patience_loss:
        return True
    return False


def clear_logging_handlers():
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def manage_saved_models(directory):  # From @Jing
    pattern = re.compile(r"epoch_(\d+)\.pth")
    epoch_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            match = pattern.match(file)
            if match:
                epoch_num = int(match.group(1))
                file_path = os.path.join(root, file)
                epoch_files.append((file_path, epoch_num))

    # Check if there are more than 5 files
    if len(epoch_files) > 1:
        epoch_files.sort(key=lambda x: x[1])
        files_to_delete = len(epoch_files) - 1

        for i in range(files_to_delete):
            os.remove(epoch_files[i][0])


def train_model(
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
):

    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(path_folder, "./loss_record.log"),
        filemode="a",
        format="%(asctime)s   %(levelname)s   %(message)s",
    )

    if optimizer_choice == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_choice == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_choice == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Invalid optimizer choice. Please choose 'Adam' or 'SGD'.")

    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

    # Initialize lists to store loss values and validation predictions
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_val_outputs = None
    best_val_labels = None
    logging.info("Training starts!")

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        for images, numerical_data, targets in train_loader:
            # Move data to GPU
            images = images.to(device)
            numerical_data = numerical_data.to(device)
            targets = targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, numerical_data).squeeze(-1)
            loss = criterion(outputs, targets)  # Ensure target shape matches output
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss}")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_outputs = []  # Temporary list to store outputs for this epoch
        val_labels = []  # Temporary list to store labels for this epoch
        with torch.no_grad():
            for images, numerical_data, targets in val_loader:
                # Move data to GPU
                images = images.to(device)
                numerical_data = numerical_data.to(device)
                targets = targets.to(device)

                # Get model output
                outputs = model(images, numerical_data).squeeze(-1)

                # Calculate loss
                loss = criterion(outputs, targets)  # Ensure target shape matches output
                val_loss += loss.item()

                # Append outputs and targets to lists
                val_outputs.append(outputs.cpu())  # Move to CPU for concatenation
                val_labels.append(targets.cpu())

        # Concatenate outputs and labels across all batches to ensure all samples are included
        val_outputs = torch.cat(val_outputs, dim=0).numpy()
        val_labels = torch.cat(val_labels, dim=0).numpy()

        # Calculate average validation loss for the epoch
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}")

        # Check if this is the best validation loss so far and store best outputs/labels
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_outputs = val_outputs
            best_val_labels = val_labels

            best_model = model
            best_model_state = model.state_dict()
            torch.save(
                best_model_state, os.path.join(path_folder, f"./epoch_{epoch + 1}.pth")
            )

        manage_saved_models(path_folder)

        logging.info(
            f"\n"
            f"Epoch:[{epoch+1}]\t Train loss={avg_train_loss:.12f}.\n"
            f"Epoch:[{epoch+1}]\t Valid loss={avg_val_loss:.12f}.\n"
        )

        if early_stopping(val_losses, patience_epochs, patience_loss):
            logging.info(f"Early stopping at epoch {epoch+1}")
            return (
                best_val_outputs,
                best_val_labels,
                best_model,
                train_losses,
                val_losses,
            )
        scheduler.step()

    logging.info("Training is done!")
    # Clear logging handlers to close the log file properly
    clear_logging_handlers()

    return best_val_outputs, best_val_labels, model, train_losses, val_losses


def test_model(model, test_loader, criterion, device):

    with torch.no_grad():  # Disable gradient calculation for inference

        test_outputs = []
        test_loss = 0.0

        for images, numerical_data, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            numerical_data = numerical_data.to(device)

            outputs = model(images, numerical_data).squeeze(-1)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # Append outputs to the list
            test_outputs.append(outputs)

        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss}")

        test_outputs = torch.cat(test_outputs, dim=0)
        test_outputs = test_outputs.cpu()
        test_outputs = test_outputs.numpy()

    return test_outputs, avg_test_loss
