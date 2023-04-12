import numpy as np
from src import run
import os
import json


datasets = {
    x: os.path.join('data', x)
    for x in os.listdir('data') if x.endswith('.csv')
}


def test_predict_mode():
    goals = {
      'ivy-league.csv': {1.0: 0.5},
      'majority-rule.csv': {1.0: 0.6},
      'circles-easy.csv': {1.0: 0.6},
      'blobs.csv': {1.0: 0.55},
    }

    learner_type = 'predict_mode'
    for key in goals:
        for fraction, goal in goals[key].items():
            accuracy = run(datasets.get(key), learner_type, fraction)
            message = f"On {key} dataset with fraction of {fraction}, "
            message += f"expected {goal:.2f} but got {accuracy:.3f}"
            assert accuracy >= goal, message


def test_comparisons():
    comparisons = [
        # predict_mode beats decision tree on majority rule w/ 0.7 frac
        ('majority-rule.csv', 0.7, 'predict_mode', 'decision_tree',
         {}, {}),

        # decision tree beats predict mode on majority rule w/ 1.0 frac
        ('majority-rule.csv', 1.0, 'decision_tree', 'predict_mode',
         {}, {}),
    ]

    for i, comparison in enumerate(comparisons):
        (key, fraction, method_a, method_b,
         kwargs_a, kwargs_b) = comparison

        data_path = datasets.get(key)
        acc_a = run(data_path, method_a, fraction, **kwargs_a)
        acc_b = run(data_path, method_b, fraction, **kwargs_b)

        if method_a != method_b:
            a, b = method_a, method_b
        else:
            a, b = kwargs_a, kwargs_b
        message = f"{a} should beat {b} in comparion #{i + 1}"
        message += f" on {key}, but {acc_a:.3f} <= {acc_b:.3f}"
        assert acc_a > acc_b, message
