import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiInputModel(nn.Module):
    def __init__(
        self,
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
    ):
        super(MultiInputModel, self).__init__()
        self.activation_cnn = activation_cnn
        self.activation_numerical = activation_numerical
        self.activation_final = activation_final
        self.dropout_rate = dropout_rate

        self.convs = nn.ModuleList()

        for feature in features_cnn:
            self.convs.extend(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=feature,
                        kernel_size=kernel_size,
                        padding=1,
                        stride=stride,
                    ),
                    self.activation_cnn,
                    nn.MaxPool2d(kernel_size=2),
                    nn.Dropout(self.dropout_rate),  # Dropout after convolutional blocks
                ]
            )
            in_channels = feature

        # Calculate the size of the flattened feature map after CNN layers
        self.flattened_size = self._get_flattened_size(image_height, image_width)

        # Fully Connected layer after CNN
        self.fc_cnn = nn.Linear(self.flattened_size, 16)
        self.dropout_cnn = nn.Dropout(self.dropout_rate)  # Dropout after CNN FC layer

        # Fully Connected layers for numerical input
        self.linear_step = nn.ModuleList()

        for feature_numerical in features_numerical:
            self.linear_step.extend(
                [
                    nn.Linear(
                        in_features=num_numerical_inputs, out_features=feature_numerical
                    ),
                    self.activation_numerical,
                    nn.Dropout(self.dropout_rate),  # Dropout in numerical branch
                ]
            )
            num_numerical_inputs = feature_numerical

        # Final Layers after Concatenation
        self.fc_combined1 = nn.Linear(16 + num_numerical_inputs, 4)
        self.dropout_combined = nn.Dropout(
            self.dropout_rate
        )  # Dropout after combined FC layer
        self.output = nn.Linear(4, 1)  # Output a single value

    def _get_flattened_size(self, height, width):
        # Simulate passing an input through the CNN to get the output size
        x = torch.zeros(1, self.convs[0].in_channels, height, width)
        for conv in self.convs:
            x = conv(x)
        flattened_size = x.view(1, -1).size(1)
        return flattened_size

    def forward(self, image, numerical):
        # CNN Branch
        x = image
        for conv in self.convs:
            x = conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.activation_cnn(self.fc_cnn(x))
        x = self.dropout_cnn(x)  # Dropout after CNN FC layer

        # MLP Branch
        y = numerical
        for linear in self.linear_step:
            y = linear(y)

        # Concatenate the outputs
        combined = torch.cat((x, y), dim=1)

        # Final Layers
        z = self.activation_final(self.fc_combined1(combined))
        z = self.dropout_combined(z)  # Dropout after combined FC layer
        output = self.output(z)  # Single numerical output

        return output
