import torch
from pytorch_lightning.metrics import Metric

class Accuracy(Metric):
    """
    Accuracy Metric: This class computes the accuracy by comparing predictions
    with the ground truth. It is used for classification tasks.
    """
    def __init__(self, dist_sync_on_step=False):
        """
        Initializes the Accuracy metric by setting up state variables to track
        the correct predictions and the total number of targets.

        Args:
            dist_sync_on_step (bool): Whether to synchronize the metric state across
                                       distributed workers during training.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        """
        Updates the accuracy metric with the current batch's predictions and targets.

        Args:
            logits (tensor): The model's predictions (raw outputs before softmax).
            target (tensor): The ground truth labels.
        """
        # Detach the logits and target to avoid gradient tracking and ensure correct device placement
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        
        # Get predictions by finding the index of the maximum logit
        preds = logits.argmax(dim=-1)

        # Filter out targets that are -100 (used for ignoring labels during loss calculation)
        preds = preds[target != -100]
        target = target[target != -100]

        if target.numel() == 0:  # If no valid targets are found, return 1 as a default
            return 1

        assert preds.shape == target.shape  # Ensure that predictions and targets have the same shape

        # Update correct and total counts
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        """Returns the computed accuracy: correct / total."""
        return self.correct / self.total


class Scalar(Metric):
    """
    Scalar Metric: This class computes the average of scalar values across batches.
    Useful for tracking metrics such as loss or any continuous scalar value.
    """
    def __init__(self, dist_sync_on_step=False):
        """
        Initializes the Scalar metric by setting up state variables to accumulate
        scalar values and their counts.

        Args:
            dist_sync_on_step (bool): Whether to synchronize the metric state across
                                       distributed workers during training.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        """
        Updates the scalar metric with a new scalar value (such as loss or another metric).

        Args:
            scalar (float or tensor): The scalar value to update the metric with.
        """
        # Ensure scalar is a tensor on the correct device
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)

        # Accumulate the scalar value and total count
        self.scalar += scalar
        self.total += 1

    def compute(self):
        """Returns the average of the scalar values: sum of scalars / total count."""
        return self.scalar / self.total


class VQAScore(Metric):
    """
    VQA (Visual Question Answering) Score Metric: This class computes the score for a VQA task.
    The VQA score is computed based on the accuracy of the answers to visual questions.
    """
    def __init__(self, dist_sync_on_step=False):
        """
        Initializes the VQA Score metric by setting up state variables to accumulate
        the correct scores and the total number of answers.

        Args:
            dist_sync_on_step (bool): Whether to synchronize the metric state across
                                       distributed workers during training.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        """
        Updates the VQA Score with the current batch's predictions and ground truth answers.

        Args:
            logits (tensor): The model's raw outputs for each possible answer (before softmax).
            target (tensor): The ground truth labels (one-hot encoded answers).
        """
        # Detach logits and target and move them to the correct device
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        
        # Get the predicted class (index of the max logit)
        logits = torch.max(logits, 1)[1]

        # Convert target to one-hot encoding based on predicted class
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)

        # Compute the score by multiplying one-hot encoded predictions with ground truth
        scores = one_hots * target

        # Update total score and total count
        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        """Returns the computed VQA score: sum of correct predictions / total samples."""
        return self.score / self.total
