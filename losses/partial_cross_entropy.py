import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss calculated only on pixels
    where a label is available.
    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions, labels):
        """
        predictions: model output
                     [batch, classes, height, width]

        labels:      partial labels
                     [batch, height, width]

        Unlabelled pixels should have label = -1.
        """

        # Find pixels that have a valid label
        valid_pixels = labels != -1

        # If there are no labelled pixels, return zero loss
        if valid_pixels.sum() == 0:
            return predictions.sum() * 0.0

        # Select only labelled pixels
        selected_predictions = predictions.permute(0, 2, 3, 1)[valid_pixels]
        selected_labels = labels[valid_pixels]

        # Calculate cross entropy only for labelled pixels
        loss = F.cross_entropy(
            selected_predictions,
            selected_labels
        )

        return loss