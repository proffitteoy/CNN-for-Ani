import torch

from cnn_for_ani.training import make_stratified_folds, should_stop_early


def test_stratified_folds_are_deterministic_complete_and_disjoint() -> None:
    labels = torch.tensor(
        [
            [index % 10, (index // 2) % 10, (index // 3) % 10, (index // 5) % 10]
            for index in range(103)
        ]
    )

    folds = make_stratified_folds(labels, fold_count=5, seed=123)

    assert folds == make_stratified_folds(labels, fold_count=5, seed=123)
    assert sorted(index for fold in folds for index in fold) == list(range(len(labels)))
    assert max(map(len, folds)) - min(map(len, folds)) <= 1
    overlaps = (
        len(set(folds[left]) & set(folds[right])) for left in range(5) for right in range(left)
    )
    assert sum(overlaps) == 0


def test_stratified_folds_balance_position_margins() -> None:
    labels = torch.tensor([[digit, digit, digit, digit] for digit in range(10) for _ in range(10)])

    folds = make_stratified_folds(labels, fold_count=5, seed=123)

    for fold in folds:
        fold_labels = labels[fold]
        for position in range(4):
            counts = torch.bincount(fold_labels[:, position], minlength=10)
            assert counts.min().item() >= 1
            assert counts.max().item() <= 3


def test_early_stopping_cannot_fire_before_minimum_epochs() -> None:
    assert not should_stop_early(
        epoch=99,
        min_epochs=100,
        epochs_without_improvement=500,
        patience=50,
    )
    assert not should_stop_early(
        epoch=100,
        min_epochs=100,
        epochs_without_improvement=49,
        patience=50,
    )
    assert should_stop_early(
        epoch=100,
        min_epochs=100,
        epochs_without_improvement=50,
        patience=50,
    )
