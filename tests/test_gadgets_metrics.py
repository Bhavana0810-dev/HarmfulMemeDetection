import pytest
import torch
from torch import nn
from .metrics import Accuracy, Scalar, VQAScore  # Assuming your metrics are in metrics.py

@pytest.fixture
def logits_and_targets():
    """
    Fixture to provide dummy logits and targets for testing.
    """
    logits = torch.tensor([[2.0, 1.0], [0.5, 2.5], [1.0, 1.0]])  # 3 samples, 2 classes
    target = torch.tensor([0, 1, 0])  # Ground truth labels
    return logits, target

def test_accuracy_update(logits_and_targets):
    """
    Test Accuracy metric update with a batch of logits and targets.
    """
    logits, target = logits_and_targets
    accuracy = Accuracy()
    
    # Update the metric with the batch
    accuracy.update(logits, target)
    
    # Compute the accuracy and check
    computed_accuracy = accuracy.compute()
    
    # Expected accuracy (2 out of 3 correct predictions)
    expected_accuracy = 2 / 3
    assert computed_accuracy == expected_accuracy, f"Expected {expected_accuracy}, but got {computed_accuracy}"

def test_accuracy_no_valid_targets():
    """
    Test Accuracy when there are no valid targets (target contains -100).
    """
    logits = torch.tensor([[2.0, 1.0], [0.5, 2.5], [1.0, 1.0]])  # 3 samples, 2 classes
    target = torch.tensor([-100, -100, -100])  # No valid targets
    
    accuracy = Accuracy()
    
    # Update the metric
    accuracy.update(logits, target)
    
    # Compute accuracy, which should be 1 as no valid targets were available
    computed_accuracy = accuracy.compute()
    
    assert computed_accuracy == 1, f"Expected 1, but got {computed_accuracy}"

def test_scalar_update():
    """
    Test Scalar metric update and compute.
    """
    scalar_metric = Scalar()
    
    # Update with some scalar values
    scalar_metric.update(1.5)
    scalar_metric.update(2.0)
    
    # Compute average
    computed_average = scalar_metric.compute()
    
    # Expected average (1.5 + 2.0) / 2 = 1.75
    expected_average = 1.75
    assert computed_average == expected_average, f"Expected {expected_average}, but got {computed_average}"

def test_vqa_score_update(logits_and_targets):
    """
    Test VQA Score update with logits (predictions) and one-hot target labels.
    """
    logits, target = logits_and_targets
    vqa_score = VQAScore()
    
    # Update the metric with the batch
    vqa_score.update(logits, target)
    
    # Compute the score
    computed_score = vqa_score.compute()
    
    # Expected score should be 2 correct answers out of 3 samples
    expected_score = 2 / 3
    assert computed_score == expected_score, f"Expected {expected_score}, but got {computed_score}"

def test_vqa_score_edge_case():
    """
    Test VQA Score when all predictions are wrong.
    """
    logits = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])  # 3 wrong predictions
    target = torch.tensor([0, 0, 0])  # Ground truth labels (all wrong predictions)
    
    vqa_score = VQAScore()
    
    # Update the metric with the batch
    vqa_score.update(logits, target)
    
    # Compute the score
    computed_score = vqa_score.compute()
    
    # Since all predictions are wrong, the score should be 0
    expected_score = 0
    assert computed_score == expected_score, f"Expected {expected_score}, but got {computed_score}"

def test_vqa_score_with_no_valid_targets():
    """
    Test VQA Score when there are no valid targets (target contains -100).
    """
    logits = torch.tensor([[2.0, 1.0], [0.5, 2.5], [1.0, 1.0]])  # 3 samples, 2 classes
    target = torch.tensor([-100, -100, -100])  # No valid targets
    
    vqa_score = VQAScore()
    
    # Update the metric
    vqa_score.update(logits, target)
    
    # Compute VQA score, which should be 0 as no valid targets were available
    computed_score = vqa_score.compute()
    
    assert computed_score == 0, f"Expected 0, but got {computed_score}"

# Run the tests
if __name__ == "__main__":
    pytest.main()
