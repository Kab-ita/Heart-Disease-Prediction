from django.shortcuts import render, redirect
from django.http import Http404
from .forms import PatientForm
from .tree_logic import (
    run_predictions_pipeline, ID3_TREE, C4_5_TREE, render_tree_to_html,
    AGREEMENT_STATS, METRICS_REPORT, render_tree_diagram_bare,
)
from .models import Patient, Assessment


def _agreement_label(a):
    if a.rf_pred is None:
        return ("Agree", "bg-secondary") if a.models_agree else ("Disagree", "bg-warning text-dark")
    if a.unanimous:
        return "Unanimous 3/3", "bg-secondary"
    return "2-1 Split", "bg-warning text-dark"

def home(request):
    assessments = Assessment.objects.select_related("patient").all()
    total = assessments.count()
    disease = sum(1 for a in assessments if a.verdict == 1)
    no_disease = total - disease
    disagreements = sum(1 for a in assessments if not a.models_agree)
    recent_records = []
    for a in assessments[:5]:
        label, badge_class = _agreement_label(a)
        recent_records.append({
            'id': a.id,
            'patient_id': a.patient.mrn,
            'created_at': a.created_at,
            'age': a.age,
            'sex': a.sex,
            'id3_result': a.id3_pred,
            'c45_result': a.c45_pred,
            'rf_result': a.rf_pred,
            'agree': a.models_agree,
            'agreement_label': label,
            'agreement_badge_class': badge_class,
            'verdict': a.verdict,
        })
    return render(request, "predictor/home.html", {
        "total": total, "disease": disease, "no_disease": no_disease,
        "disagreements": disagreements, "recent": recent_records
    })


def predict(request):
    form = PatientForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        raw_payload = form.cleaned_data
        results = run_predictions_pipeline(raw_payload)

        patient, _ = Patient.objects.get_or_create(
            mrn=raw_payload["patient_id"],
            defaults={"name": raw_payload["patient_name"]},
        )
        id3_pred = results["id3"]["prediction"]
        c45_pred = results["c45"]["prediction"]
        rf_pred = results["rf"]["prediction"]
        vote = results["vote"]
        Assessment.objects.create(
            patient=patient,
            age=raw_payload["age"], sex=raw_payload["sex"],
            chest_pain_type=raw_payload["chest_pain_type"], resting_bp=raw_payload["resting_bp"],
            cholesterol=raw_payload["cholesterol"], fasting_bs=raw_payload["fasting_bs"],
            resting_ecg=raw_payload["resting_ecg"], max_hr=raw_payload["max_hr"],
            exercise_angina=raw_payload["exercise_angina"], oldpeak=raw_payload["oldpeak"],
            st_slope=raw_payload["st_slope"],
            id3_pred=id3_pred, c45_pred=c45_pred,
            rf_pred=rf_pred, rf_confidence=results["rf"]["confidence"],
        )
        abnormal_flags = []
        if float(raw_payload.get('oldpeak', 0)) > 1.5:
            abnormal_flags.append(f"ST depression of {raw_payload.get('oldpeak')} mm (significant indicator)")
        if int(raw_payload.get('max_hr', 150)) < 100 or int(raw_payload.get('max_hr', 150)) > 180:
            abnormal_flags.append(f"Max heart rate of {raw_payload.get('max_hr')} bpm (outside typical range)")
        if int(raw_payload.get('resting_bp', 120)) > 140:
            abnormal_flags.append(f"Resting blood pressure of {raw_payload.get('resting_bp')} mmHg (stage 2 hypertension range)")
        if int(raw_payload.get('cholesterol', 200)) > 240:
            abnormal_flags.append(f"Serum cholesterol of {raw_payload.get('cholesterol')} mg/dl (high range)")
        if not abnormal_flags:
            cp = raw_payload.get('chest_pain_type', 'TA')
            slope = raw_payload.get('st_slope', 'Up')
            cp_mapping = {'ASY': 'Asymptomatic', 'NAP': 'Non-Anginal', 'ATA': 'Atypical Angina', 'TA': 'Typical Angina'}
            cp_clean = cp_mapping.get(cp, cp)
            if cp == 'ASY':
                abnormal_flags.append(f"Patient presents with '{cp_clean}' chest pain, often clinically silent.")
            elif slope in ('Flat', 'Down'):
                abnormal_flags.append(f"ST/T wave slope is '{slope}' during stress, an atypical pattern.")
            else:
                abnormal_flags.append("No single flagged value; the verdict reflects the combination of inputs.")
        if vote is not None:
            verdict = vote["verdict"]
            agree = vote["unanimous"]
        else:
            agree = id3_pred == c45_pred
            verdict = id3_pred if agree else c45_pred
        request.session['last_result'] = {
            'patient_name': raw_payload['patient_name'],
            'patient_id': patient.mrn,
            'id3_pred': id3_pred,
            'c45_pred': c45_pred,
            'rf_pred': rf_pred,
            'rf_confidence': results['rf']['confidence'],
            'agree': agree,
            'unanimous': vote["unanimous"] if vote else None,
            'tally': vote["tally"] if vote else None,
            'verdict': verdict,
            'id3_path': results['id3']['explanation'],
            'c45_path': results['c45']['explanation'],
            'rf_path': results['rf']['explanation'],
            'id3_tree_html': results['id3']['tree_html'],
            'c45_tree_html': results['c45']['tree_html'],
            'most_abnormal': abnormal_flags[0],
        }
        return redirect('result')

    return render(request, "predictor/predict.html", {"form": form})

def result(request):
    result_data = request.session.get('last_result', None)
    if not result_data:
        return redirect('predict')
    return render(request, "predictor/result.html", {"result": result_data})


def tree_detail(request, model_name):
    if model_name not in ("id3", "c45"):
        raise Http404("Unknown model")
    last_result = request.session.get('last_result') or {}
    tree_html = last_result.get(f'{model_name}_tree_html')
    highlighted = tree_html is not None
    if not highlighted:
        tree_html = render_tree_diagram_bare(model_name)
    return render(request, "predictor/tree_detail.html", {
        "model_label": "ID3" if model_name == "id3" else "C4.5",
        "tree_html": tree_html,
        "highlighted": highlighted,
    })
def compare(request):
    FALLBACK_METRICS = {
        "id3": {"label": "Independently Implemented ID3", "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0},
        "c45": {"label": "Independently Implemented C4.5", "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0},
        "random_forest": {"label": "scikit-learn Random Forest (Reference Ensemble)", "accuracy": 0.0, "precision": 0.0, "recall":0.0, "f1_score": 0.0},
    }
    LABELS = {
        "id3": "ID3 Model",
        "c45": "C4.5 Model",
        "random_forest": "Random Forest"
    }
    report = METRICS_REPORT or {}
    metrics_are_live = METRICS_REPORT is not None
    metrics_summary = {}
    for key in ("id3", "c45", "random_forest"):
        entry = report.get(key) or FALLBACK_METRICS[key]
        display_name=LABELS[key]
        metrics_summary[display_name] = entry

    id3_tree_html = render_tree_to_html(ID3_TREE)
    c45_tree_html = render_tree_to_html(C4_5_TREE)

    chart_data = {
        "labels": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "datasets": [
            {
                "label": name,
                "values": [m["accuracy"], m["precision"], m["recall"], m["f1_score"]],
            }
            for name, m in metrics_summary.items()
        ],
    }
    return render(request, "predictor/compare.html", {
        "metrics": metrics_summary, "id3_tree_html": id3_tree_html, "c45_tree_html": c45_tree_html,
        "agreement_stats": AGREEMENT_STATS, "chart_data": chart_data, "metrics_are_live": metrics_are_live,
    })
def history(request):
    formatted_records = []
    for a in Assessment.objects.select_related("patient").all():
        label, badge_class = _agreement_label(a)
        formatted_records.append({
            'id': a.id,
            'patient_id': a.patient.mrn,
            'created_at': a.created_at, 'age': a.age, 'sex': a.sex,
            'id3_result': a.id3_pred, 'c45_result': a.c45_pred, 'rf_result': a.rf_pred,
            'agree': a.models_agree,
            'agreement_label': label,
            'agreement_badge_class': badge_class,
            'verdict': a.verdict,
        })
    return render(request, "predictor/history.html", {"records": formatted_records})