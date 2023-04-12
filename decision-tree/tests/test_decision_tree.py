import numpy as np
from src import run
import os
import json


datasets = {
    x: os.path.join('data', x)
    for x in os.listdir('data') if x.endswith('.csv')
}


def test_decision_tree_binary_predict():
    from src.decision_tree import DecisionTree, Node
    attribute_names = ['Outlook', 'Temp', 'Wind']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    root = Node(
        attribute_name="Outlook", attribute_index=0,
        split_value=0.5, branches=[])

    left = Node(
        attribute_name="Temp", attribute_index=1,
        split_value=0.5, branches=[])

    left_left = Node(
        attribute_name=None, attribute_index=None,
        return_value=1, branches=[])

    left_right = Node(
        attribute_name=None, attribute_index=None,
        return_value=0, branches=[])

    right = Node(
        attribute_name=None, attribute_index=None,
        return_value=1, branches=[])

    left.branches = [left_left, left_right]
    root.branches = [left, right]
    decision_tree.tree = root
    examples = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1]])
    predictions = decision_tree.predict(examples)
    ground_truth = np.array([[1], [0], [1], [1], [0]])

    np.testing.assert_allclose(predictions, ground_truth)


def test_decision_tree_continuous_predict():
    from src.decision_tree import DecisionTree, Node
    attribute_names = ['Wind', 'Temp', 'Outlook']
    examples = np.array([[1, 79.21, 1], [2, 90.56, 0], [7, 88.36, 1], [5, 84.02, 0], [1, 43.77, 0]])
    decision_tree = DecisionTree(attribute_names=attribute_names)
    root = Node(
        attribute_name="Wind", attribute_index=0,
        split_value=np.median(examples[:, 0]), branches=[])

    left = Node(
        attribute_name="Temp", attribute_index=1,
        split_value=np.median(examples[:, 1]), branches=[])

    left_left = Node(
        attribute_name=None, attribute_index=None,
        return_value=1, branches=[])

    left_right = Node(
        attribute_name=None, attribute_index=None,
        return_value=0, branches=[])

    right = Node(
        attribute_name=None, attribute_index=None,
        return_value=0, branches=[])

    left.branches = [left_left, left_right]
    root.branches = [left, right]
    decision_tree.tree = root

    predictions = decision_tree.predict(examples)
    ground_truth = np.array([[1], [0], [0], [0], [1]])

    np.testing.assert_allclose(predictions, ground_truth)


def test_information_gain():
    from src import load_data
    from src import information_gain

    _features, _targets, _attribute_names = load_data('data/play-tennis.csv')
    iGHumidity = information_gain(_features, 2, _targets)
    iGWind = information_gain(_features, 3, _targets)
    realIGHumidity = 0.1515
    realIGWind = 0.048

    assert np.abs(iGHumidity-realIGHumidity) < 1e-3
    assert np.abs(iGWind - realIGWind) < 1e-3


def test_decision_tree_run():
    goals = {
        'xor-easy.csv': {1.0: 1.0},
        'xor-hard.csv': {1.0: 0.8, 0.8: 1.0},
        'ivy-league.csv': {1.0: .9, 0.8: 0.6,},
        'majority-rule.csv': {1.0: 1.0, 0.8: 0.8,},
        'circles-hard.csv': {1.0: 0.7},
        'circles-easy.csv': {1.0: 0.8},
        'blobs.csv': {1.0: 0.8, 0.8: 0.9,},
    }

    order = [
        'xor-easy.csv',
        'ivy-league.csv',
        'majority-rule.csv',
        'xor-hard.csv',
        'blobs.csv',
        'circles-easy.csv',
        'circles-hard.csv',
    ]

    learner_type = 'decision_tree'
    for key in order:
        for fraction, goal in goals[key].items():
            accuracy = run(datasets.get(key), learner_type, fraction)
            message = f"On {key} dataset with fraction {fraction}, "
            message += f"expected {goal:.2f} but got {accuracy:.3f}"
            assert accuracy >= goal, message

