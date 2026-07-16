import torch

from cnn_for_ani.error_analysis import confusion_matrix


def test_confusion_matrix_uses_true_rows_and_prediction_columns() -> None:
    targets = torch.tensor([[3, 8, 3], [1, 7, 1]])
    predictions = torch.tensor([[8, 8, 3], [7, 1, 1]])

    matrix = confusion_matrix(targets, predictions)

    assert matrix[3, 8].item() == 1
    assert matrix[3, 3].item() == 1
    assert matrix[8, 8].item() == 1
    assert matrix[1, 7].item() == 1
    assert matrix[1, 1].item() == 1
    assert matrix[7, 1].item() == 1
