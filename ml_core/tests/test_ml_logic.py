import math
import pytest

from ml_core.discretize import (
    bin_age, bin_resting_bp, bin_cholesterol, bin_max_hr, bin_oldpeak, discretize_row
)
from ml_core.id3 import calculate_entropy, calculate_information_gain, build_id3_tree, predict_id3
from ml_core.c45 import _split_info, find_best_continuous_split, build_c45_tree, predict_c45
from ml_core.validation import performance_report, majority_vote
 
def test_bin_age_boundaries():
    assert bin_age(34) == "Young"
    assert bin_age(35) == "Middle-Aged"
    assert bin_age(55) == "Middle-Aged"
    assert bin_age(56) == "Senior"


def test_bin_resting_bp_boundaries():
    assert bin_resting_bp(119) == "Normal"
    assert bin_resting_bp(120) == "Pre-Hypertension"
    assert bin_resting_bp(139) == "Pre-Hypertension"
    assert bin_resting_bp(140) == "Hypertension"


def test_bin_cholesterol_boundaries():
    assert bin_cholesterol(199) == "Desirable"
    assert bin_cholesterol(200) == "Borderline"
    assert bin_cholesterol(239) == "Borderline"
    assert bin_cholesterol(240) == "High"


def test_bin_max_hr_boundaries():
    assert bin_max_hr(99) == "Low"
    assert bin_max_hr(100) == "Moderate"
    assert bin_max_hr(149) == "Moderate"
    assert bin_max_hr(150) == "High"


def test_bin_oldpeak_boundaries():
    assert bin_oldpeak(-1) == "None"
    assert bin_oldpeak(0) == "None"
    assert bin_oldpeak(0.1) == "Mild"
    assert bin_oldpeak(2) == "Mild"
    assert bin_oldpeak(2.1) == "Severe"


def test_discretize_row_full_mapping():
    row = {
        "Age": 61, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 145,
        "Cholesterol": 250, "FastingBS": 1, "RestingECG": "ST", "MaxHR": 95,
        "ExerciseAngina": "Y", "Oldpeak": 2.5, "ST_Slope": "Down"
    }
    result = discretize_row(row)
    assert result == {
        "Age": "Senior", "Sex": "M", "ChestPainType": "ASY", "RestingBP": "Hypertension",
        "Cholesterol": "High", "FastingBS": "Yes", "RestingECG": "ST", "MaxHR": "Low",
        "ExerciseAngina": "Y", "Oldpeak": "Severe", "ST_Slope": "Down"
    }


def test_discretize_row_fasting_bs_accepts_string_or_int():
    base = {"Age": 40, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 110,
            "Cholesterol": 180, "RestingECG": "Normal", "MaxHR": 160,
            "ExerciseAngina": "N", "Oldpeak": 0, "ST_Slope": "Up"}
    assert discretize_row({**base, "FastingBS": "0"})["FastingBS"] == "No"
    assert discretize_row({**base, "FastingBS": 1})["FastingBS"] == "Yes"

 
def test_entropy_empty_dataset():
    assert calculate_entropy([], 1) == 0.0


def test_entropy_pure():
    assert calculate_entropy([["M", 1], ["F", 1]], 1) == 0.0


def test_entropy_balanced():
    assert math.isclose(calculate_entropy([["M", 1], ["F", 0]], 1), 1.0)


def test_entropy_skewed():
    data = [["a", 1], ["b", 1], ["c", 1], ["d", 0]]
    assert math.isclose(calculate_entropy(data, 1), 0.8113, abs_tol=1e-3)


def test_information_gain_perfect_split():
    data = [["X", 1], ["X", 1], ["Y", 0], ["Y", 0]]
    gain = calculate_information_gain(data, 0, 1)
    assert math.isclose(gain, 1.0)


def test_build_id3_tree_pure_dataset_returns_leaf():
    data = [["X", 1], ["Y", 1]]
    tree = build_id3_tree(data, [0], 1, ["feat"])
    assert tree == 1


def test_build_id3_tree_no_features_returns_majority():
    data = [["X", 1], ["Y", 0], ["Z", 1]]
    tree = build_id3_tree(data, [], 1, ["feat"])
    assert tree == 1


def test_predict_id3_seen_and_unseen_branch():
    data = [["Young", 0], ["Senior", 1], ["Senior", 1]]
    tree = build_id3_tree(data, [0], 1, ["Age"])

    pred_seen, path_seen = predict_id3(tree, {"Age": "Senior"})
    assert pred_seen == 1
    assert "(unseen value fallback)" not in path_seen[-1]

    pred_unseen, path_unseen = predict_id3(tree, {"Age": "Middle-Aged"})
    assert "(unseen value fallback)" in path_unseen[-1]

 
def test_split_info_single_partition_returns_epsilon():
    si = _split_info([[1, 2, 3]], 3)
    assert si == 1e-9


def test_split_info_even_split():
    si = _split_info([[1, 2], [3, 4]], 4)
    assert math.isclose(si, 1.0)


def test_continuous_split_threshold():
    toy_data = [[10.0, 0], [20.0, 0], [30.0, 1], [40.0, 1]]
    _, threshold, partitions = find_best_continuous_split(toy_data, 0, 1)
    assert threshold == 25.0
    assert len(partitions["<="]) == 2
    assert len(partitions[">"]) == 2


def test_continuous_split_single_unique_value():
    toy_data = [[10.0, 0], [10.0, 1]]
    gr, threshold, partitions = find_best_continuous_split(toy_data, 0, 1)
    assert (gr, threshold, partitions) == (0.0, None, None)


def test_build_c45_tree_pure_dataset_returns_leaf():
    data = [[10.0, "A", 1], [20.0, "A", 1]]
    tree = build_c45_tree(data, [0, 1], 2, ["num", "cat"], continuous_features=["num"])
    assert tree == 1


def test_predict_c45_continuous_and_categorical_paths():
    data = [[10.0, "X", 0], [20.0, "X", 0], [30.0, "Y", 1], [40.0, "Y", 1]]
    tree = build_c45_tree(data, [0, 1], 2, ["num", "cat"], continuous_features=["num"])

    pred, path = predict_c45(tree, {"num": 15.0, "cat": "X"})
    assert isinstance(pred, int)
    assert len(path) >= 1


def test_predict_c45_unseen_categorical_falls_back():
    data = [[10.0, "X", 0], [20.0, "Y", 1]]
    tree = build_c45_tree(data, [1], 2, ["num", "cat"], continuous_features=[])
    if isinstance(tree, dict) and tree["mode"] == "categorical":
        pred, path = predict_c45(tree, {"cat": "Z"})
        assert "(fallback)" in path[-1]

 

def test_performance_report_perfect_predictions():
    report = performance_report([1, 0, 1, 0], [1, 0, 1, 0])
    assert report["accuracy"] == 100.0
    assert report["precision"] == 100.0
    assert report["recall"] == 100.0
    assert report["f1_score"] == 100.0


def test_performance_report_all_wrong():
    report = performance_report([1, 1, 0, 0], [0, 0, 1, 1])
    assert report["accuracy"] == 0.0


def test_performance_report_confusion_matrix_counts():
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0]
    report = performance_report(y_true, y_pred)
    cm = report["confusion_matrix"]
    assert cm == {"TP": 2, "FP": 1, "FN": 1, "TN": 2}


def test_performance_report_empty_input_no_crash():
    report = performance_report([], [])
    assert report["accuracy"] == 0
    assert report["confusion_matrix"] == {"TP": 0, "FP": 0, "FN": 0, "TN": 0}


def test_majority_vote_unanimous():
    result = majority_vote(1, 1, 1)
    assert result["verdict"] == 1
    assert result["unanimous"] is True
    assert result["tally"] == {0: 0, 1: 3}


def test_majority_vote_two_to_one():
    result = majority_vote(1, 0, 1)
    assert result["verdict"] == 1
    assert result["unanimous"] is False
    assert result["tally"] == {0: 1, 1: 2}


def test_majority_vote_rf_is_deciding_vote():
   
    assert majority_vote(1, 0, 1)["verdict"] == 1
    assert majority_vote(1, 0, 0)["verdict"] == 0


def test_majority_vote_votes_dict_matches_inputs():
    result = majority_vote(0, 1, 1)
    assert result["votes"] == {"id3": 0, "c45": 1, "random_forest": 1}