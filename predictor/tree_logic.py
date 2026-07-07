import json
import os
import joblib
from django.conf import settings
from ml_core.discretize import discretize_row
from ml_core.id3 import predict_id3
from ml_core.c45 import predict_c45
from ml_core.random_forest import predict_rf
from ml_core.validation import majority_vote

MODELS_DIR = os.path.join(settings.BASE_DIR, 'ml_models')

with open(os.path.join(MODELS_DIR, 'id3_tree.json')) as f:
    ID3_TREE = json.load(f)
with open(os.path.join(MODELS_DIR, 'c45_tree.json')) as f:
    C4_5_TREE = json.load(f)

try:
    with open(os.path.join(MODELS_DIR, 'agreement_stats.json')) as f:
        AGREEMENT_STATS = json.load(f)
except FileNotFoundError:
    AGREEMENT_STATS = None

try:
    with open(os.path.join(MODELS_DIR, 'metrics_report.json')) as f:
        METRICS_REPORT = json.load(f)
except FileNotFoundError:
    METRICS_REPORT = None

try:
    RF_BUNDLE = joblib.load(os.path.join(MODELS_DIR, 'rf_model.joblib'))
except FileNotFoundError:
    RF_BUNDLE = None
def run_predictions_pipeline(raw_input):
    cased_payload = {
        "Age": raw_input["age"],
        "Sex": raw_input["sex"],
        "ChestPainType": raw_input["chest_pain_type"],
        "RestingBP": raw_input["resting_bp"],
        "Cholesterol": raw_input["cholesterol"],
        "FastingBS": int(raw_input["fasting_bs"]),
        "RestingECG": raw_input["resting_ecg"],
        "MaxHR": raw_input["max_hr"],
        "ExerciseAngina": raw_input["exercise_angina"],
        "Oldpeak": raw_input["oldpeak"],
        "ST_Slope": raw_input["st_slope"]
    }

    binned_data = discretize_row(cased_payload)
    id3_out, id3_path = predict_id3(ID3_TREE, binned_data)
    c45_out, c45_path = predict_c45(C4_5_TREE, cased_payload)
    id3_tree_html = render_id3_tree_with_path(binned_data)
    c45_tree_html = render_c45_tree_with_path(cased_payload)

    if RF_BUNDLE is not None:
        rf_out, rf_path, rf_confidence = predict_rf(
            RF_BUNDLE["model"], RF_BUNDLE["encoders"], RF_BUNDLE["features"], cased_payload
        )
        vote = majority_vote(id3_out, c45_out, rf_out)
    else: 
        rf_out, rf_path, rf_confidence, vote = None, [], None, None

    return {
        "id3": {"prediction": id3_out, "explanation": id3_path, "tree_html": id3_tree_html},
        "c45": {"prediction": c45_out, "explanation": c45_path, "tree_html": c45_tree_html},
        "rf": {"prediction": rf_out, "explanation": rf_path, "confidence": rf_confidence},
        "vote": vote,  
    }


def render_tree_to_html(tree):
    if not isinstance(tree, dict):
        badge_style = "background: linear-gradient(135deg, #d90429, #ef233c); color:white;" if tree == 1 else "background: linear-gradient(135deg, #198754, #2ec4b6); color:white;"
        label = "Heart Disease (1)" if tree == 1 else "Normal Baseline (0)"
        return f'<span class="badge px-3 py-2 fw-bold shadow-sm rounded-pill" style="{badge_style}">{label}</span>'

    feature_name = tree.get("feature")
    tree_type = tree.get("type", "c45")
    threshold = tree.get("threshold", "")

    html = f"<div class='tree-node-box shadow-sm d-inline-block p-2 my-1 rounded border bg-white'>"
    html += f"Feature: <span class='text-dark fw-bold'>{feature_name}</span>"
    if tree_type == "c45" and threshold != "":
        html += f" <span class='badge bg-secondary ms-1' style='font-size:0.75rem;'>Split threshold: &le; {round(threshold, 2)}</span>"
    html += "</div>"

    html += "<ul class='tree-nested-list' style='padding-left:24px; border-left:2px dashed #94a3b8; margin-left:15px; list-style-type:none;'>"
    for branch_name, sub_tree in tree["branches"].items():
        html += f"<li class='my-2'>"
        html += f"<span class='text-crimson fw-bold me-1' style='color:#8b0000;'>&rarr; [{branch_name}]:</span> "
        html += render_tree_to_html(sub_tree)
        html += "</li>"
    html += "</ul>"

    return html


def _diagram_leaf_box(value, active):
    box_style = "background: linear-gradient(135deg, #d90429, #ef233c); color:white;" if value == 1 else "background: linear-gradient(135deg, #198754, #2ec4b6); color:white;"
    label = "Heart Disease (1)" if value == 1 else "Normal Baseline (0)"
    active_class = " diagram-leaf-active" if active else ""
    return f'<div class="diagram-box diagram-leaf{active_class}" style="{box_style}">{label}</div>'


def _node_summary_html(tree, active):
    feature_name = tree.get("feature")
    tree_type = tree.get("type", "c45")
    threshold = tree.get("threshold", "")

    html = f'<span class="outline-feature{" outline-active" if active else ""}">{feature_name}</span>'
    if tree_type == "c45" and threshold != "":
        html += f'<span class="outline-threshold">&le; {round(threshold, 2)}</span>'
    return html


def _node_children_html(tree, sample, active):
  
    feature_name = tree.get("feature")
    tree_type = tree.get("type", "c45")
    threshold = tree.get("threshold", "")
    taken_key = None
    if active and sample is not None:
        if tree_type == "c45" and tree.get("mode") == "continuous":
            try:
                val = float(sample.get(feature_name))
                taken_key = "<=" if val <= threshold else ">"
            except (TypeError, ValueError):
                taken_key = None
        else:
            val = str(sample.get(feature_name))
            taken_key = val if val in tree["branches"] else None

    html = "<ul class='outline-children'>"
    for branch_name, sub_tree in tree["branches"].items():
        branch_active = active and branch_name == taken_key
        li_class = " class='outline-branch-active'" if branch_active else ""
        label_class = "outline-branch-label outline-branch-active-label" if branch_active else "outline-branch-label"

        html += f"<li{li_class}>"
        html += f"<span class='{label_class}'>{branch_name}</span>"
        if isinstance(sub_tree, dict):
            open_attr = " open" if branch_active else ""
            html += f"<details{open_attr}>"
            html += f"<summary>{_node_summary_html(sub_tree, branch_active)}</summary>"
            html += _node_children_html(sub_tree, sample, branch_active)
            html += "</details>"
        else:
            html += _diagram_leaf_box(sub_tree, branch_active)
        html += "</li>"
    html += "</ul>"
    return html


def _render_tree_outline(tree, sample, active):
    html = f'<div class="outline-root">{_node_summary_html(tree, active)}</div>'
    html += _node_children_html(tree, sample, active)
    return f'<div class="tree-outline">{html}</div>'


def render_id3_tree_with_path(binned_sample): 
    return _render_tree_outline(ID3_TREE, binned_sample, active=True)


def render_c45_tree_with_path(raw_sample):
    return _render_tree_outline(C4_5_TREE, raw_sample, active=True)


def render_tree_diagram_bare(model_name):
    tree = ID3_TREE if model_name == "id3" else C4_5_TREE
    return _render_tree_outline(tree, None, active=False)