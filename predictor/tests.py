from django.test import TestCase, Client
from django.urls import reverse
from predictor.models import Patient, Assessment


VALID_LOW_RISK = {
    "patient_name": "Alice Example", "patient_id": "P-0001",
    "age": 30, "sex": "F", "chest_pain_type": "ATA",
    "resting_bp": 110, "cholesterol": 180, "fasting_bs": 0,
    "resting_ecg": "Normal", "max_hr": 175, "exercise_angina": "N",
    "oldpeak": 0.0, "st_slope": "Up",
}

VALID_HIGH_RISK = {
    "patient_name": "Bob Example", "patient_id": "P-0002",
    "age": 65, "sex": "M", "chest_pain_type": "ASY",
    "resting_bp": 155, "cholesterol": 270, "fasting_bs": 1,
    "resting_ecg": "ST", "max_hr": 100, "exercise_angina": "Y",
    "oldpeak": 2.5, "st_slope": "Flat",
}


class TestPredictWorkflow(TestCase):
    def setUp(self):
        self.client = Client()
        self.predict_url = reverse("predict")

    def test_get_request_returns_form(self):
        response = self.client.get(self.predict_url)
        self.assertEqual(response.status_code, 200)
        self.assertIn("form", response.context)

    def test_valid_low_risk_submission_creates_patient_and_assessment(self):
        self.assertEqual(Patient.objects.count(), 0)
        self.assertEqual(Assessment.objects.count(), 0)
        response = self.client.post(self.predict_url, data=VALID_LOW_RISK)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Patient.objects.count(), 1)
        self.assertEqual(Assessment.objects.count(), 1)

    def test_valid_high_risk_submission_creates_patient_and_assessment(self):
        response = self.client.post(self.predict_url, data=VALID_HIGH_RISK)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Patient.objects.count(), 1)
        self.assertEqual(Assessment.objects.count(), 1)

    def test_out_of_range_age_rejected(self):
        bad = dict(VALID_LOW_RISK)
        bad["age"] = 999
        response = self.client.post(self.predict_url, data=bad)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Assessment.objects.count(), 0)
        self.assertTrue(response.context["form"].errors)

    def test_negative_cholesterol_rejected(self):
        bad = dict(VALID_LOW_RISK)
        bad["cholesterol"] = -50
        response = self.client.post(self.predict_url, data=bad)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Assessment.objects.count(), 0)

    def test_missing_required_field_rejected(self):
        bad = dict(VALID_LOW_RISK)
        del bad["st_slope"]
        response = self.client.post(self.predict_url, data=bad)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Assessment.objects.count(), 0)

    def test_rf_field_present_on_assessment(self):
        # RF may or may not have a trained bundle on disk in this test
        # environment, but the *field* must exist either way -- unlike the
        # old design, which had no column for it at all.
        self.client.post(self.predict_url, data=VALID_LOW_RISK)
        assessment = Assessment.objects.first()
        self.assertTrue(hasattr(assessment, "rf_pred"))

    # --- Patient identity: name + MRN ---------------------------------------

    def test_missing_patient_name_rejected(self):
        bad = dict(VALID_LOW_RISK)
        del bad["patient_name"]
        response = self.client.post(self.predict_url, data=bad)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Assessment.objects.count(), 0)
        self.assertIn("patient_name", response.context["form"].errors)

    def test_missing_patient_id_rejected(self):
        bad = dict(VALID_LOW_RISK)
        del bad["patient_id"]
        response = self.client.post(self.predict_url, data=bad)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Assessment.objects.count(), 0)
        self.assertIn("patient_id", response.context["form"].errors)

    def test_blank_patient_id_rejected(self):
        bad = dict(VALID_LOW_RISK)
        bad["patient_id"] = "   "
        response = self.client.post(self.predict_url, data=bad)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Assessment.objects.count(), 0)
        self.assertIn("patient_id", response.context["form"].errors)

    # --- Repeat patients are now expected, not rejected ---------------------

    def test_returning_patient_gets_second_assessment_not_rejected(self):
        self.client.post(self.predict_url, data=VALID_LOW_RISK)
        second_visit = dict(VALID_LOW_RISK)  # same patient, new visit
        second_visit["oldpeak"] = 1.2
        response = self.client.post(self.predict_url, data=second_visit)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Patient.objects.count(), 1)      # still one patient...
        self.assertEqual(Assessment.objects.count(), 2)   # ...with two visits

    def test_patient_id_normalized_case_and_whitespace(self):
        variant = dict(VALID_LOW_RISK)
        variant["patient_id"] = f"  {VALID_LOW_RISK['patient_id'].lower()}  "
        self.client.post(self.predict_url, data=VALID_LOW_RISK)
        response = self.client.post(self.predict_url, data=variant)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Patient.objects.count(), 1)  # normalizes to the same MRN
        self.assertEqual(Assessment.objects.count(), 2)

    def test_mismatched_name_for_existing_patient_id_rejected(self):
        self.client.post(self.predict_url, data=VALID_LOW_RISK)
        mismatch = dict(VALID_LOW_RISK)
        mismatch["patient_name"] = "A Totally Different Person"
        response = self.client.post(self.predict_url, data=mismatch)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Assessment.objects.count(), 1)  # second submission rejected
        self.assertIn("patient_name", response.context["form"].errors)

    def test_same_mrn_at_db_level_raises_integrity_error(self):
        Patient.objects.create(mrn="DUP-ID", name="First")
        from django.db import IntegrityError
        with self.assertRaises(IntegrityError):
            Patient.objects.create(mrn="DUP-ID", name="Second")


class TestResultView(TestCase):
    def setUp(self):
        self.client = Client()

    def test_result_redirects_to_predict_without_session_data(self):
        response = self.client.get(reverse("result"))
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("predict"))

    def test_result_shows_id3_and_c45_paths_after_prediction(self):
        self.client.post(reverse("predict"), data=VALID_HIGH_RISK)
        response = self.client.get(reverse("result"))
        self.assertEqual(response.status_code, 200)
        r = response.context["result"]
        self.assertIn("id3_path", r)
        self.assertIn("c45_path", r)
        self.assertIn("rf_path", r)
        self.assertTrue(len(r["id3_path"]) >= 1)
        self.assertTrue(len(r["c45_path"]) >= 1)

    def test_verdict_is_binary(self):
        self.client.post(reverse("predict"), data=VALID_LOW_RISK)
        response = self.client.get(reverse("result"))
        self.assertIn(response.context["result"]["verdict"], (0, 1))

    def test_verdict_is_majority_of_three_when_rf_available(self):
        self.client.post(reverse("predict"), data=VALID_HIGH_RISK)
        response = self.client.get(reverse("result"))
        r = response.context["result"]
        if r.get("rf_pred") is not None:
            votes = [r["id3_pred"], r["c45_pred"], r["rf_pred"]]
            expected = 1 if votes.count(1) > votes.count(0) else 0
            self.assertEqual(r["verdict"], expected)

    def test_result_page_links_to_full_decision_trees(self):
        self.client.post(reverse("predict"), data=VALID_HIGH_RISK)
        response = self.client.get(reverse("result"))
        html = response.content.decode()
        self.assertIn(reverse("tree_detail", args=["id3"]), html)
        self.assertIn(reverse("tree_detail", args=["c45"]), html)
        self.assertIn('target="_blank"', html)

    def test_result_page_shows_patient_name_and_id(self):
        self.client.post(reverse("predict"), data=VALID_HIGH_RISK)
        response = self.client.get(reverse("result"))
        html = response.content.decode()
        self.assertIn(VALID_HIGH_RISK["patient_name"], html)
        self.assertIn(VALID_HIGH_RISK["patient_id"], html)


class TestTreeDetailView(TestCase):
    def test_tree_detail_without_prediction_renders_bare_tree(self):
        response = self.client.get(reverse("tree_detail", args=["id3"]))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context["highlighted"])

    def test_tree_detail_after_prediction_is_highlighted(self):
        self.client.post(reverse("predict"), data=VALID_HIGH_RISK)
        response = self.client.get(reverse("tree_detail", args=["c45"]))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["highlighted"])
        self.assertIn("diagram-node-active", response.content.decode())

    def test_tree_detail_rejects_unknown_model(self):
        # Random Forest deliberately has no single-tree detail page -- see
        # views.tree_detail's docstring on why.
        response = self.client.get(reverse("tree_detail", args=["random_forest"]))
        self.assertEqual(response.status_code, 404)


class TestCompareView(TestCase):
    def test_compare_page_includes_all_models_and_majority_vote(self):
        response = self.client.get(reverse("compare"))
        self.assertEqual(response.status_code, 200)
        metrics = response.context["metrics"]
        self.assertEqual(len(metrics), 4)
        joined_names = " ".join(metrics.keys())
        self.assertIn("ID3", joined_names)
        self.assertIn("C4.5", joined_names)
        self.assertIn("Random Forest", joined_names)
        self.assertIn("Majority Vote", joined_names)
        self.assertIn("id3_tree_html", response.context)
        self.assertIn("c45_tree_html", response.context)

    def test_compare_page_has_chart_data(self):
        response = self.client.get(reverse("compare"))
        self.assertIn("chart_data", response.context)
        data = response.context["chart_data"]
        self.assertEqual(len(data["datasets"]), 4)
        html = response.content.decode()
        self.assertIn("metricsChart", html)
        self.assertIn('id="chart-data"', html)
        self.assertIn('type="application/json"', html)

    def test_compare_page_includes_confusion_matrices_for_every_model(self):
        response = self.client.get(reverse("compare"))
        metrics = response.context["metrics"]
        for name, m in metrics.items():
            self.assertIn("confusion_matrix", m, f"{name} is missing a confusion matrix")
            for key in ("TP", "FP", "FN", "TN"):
                self.assertIn(key, m["confusion_matrix"])
        html = response.content.decode()
        self.assertIn("Confusion Matrices", html)

    def test_compare_page_shows_majority_vote_highlight_card(self):
        response = self.client.get(reverse("compare"))
        self.assertIn("majority_vote_metrics", response.context)
        mv = response.context["majority_vote_metrics"]
        for key in ("accuracy", "precision", "recall", "f1_score"):
            self.assertIn(key, mv)


class TestHistoryView(TestCase):
    def test_history_empty_state(self):
        response = self.client.get(reverse("history"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context["records"]), 0)

    def test_history_lists_submitted_record_with_agreement_flag(self):
        self.client.post(reverse("predict"), data=VALID_HIGH_RISK)
        response = self.client.get(reverse("history"))
        self.assertEqual(len(response.context["records"]), 1)
        self.assertIn("agree", response.context["records"][0])

    def test_history_shows_patient_id_but_never_name(self):
        self.client.post(reverse("predict"), data=VALID_HIGH_RISK)
        response = self.client.get(reverse("history"))

        record_ctx = response.context["records"][0]
        self.assertEqual(record_ctx["patient_id"], VALID_HIGH_RISK["patient_id"])
        self.assertNotIn("patient_name", record_ctx)

        html = response.content.decode()
        self.assertIn(VALID_HIGH_RISK["patient_id"], html)
        self.assertNotIn(VALID_HIGH_RISK["patient_name"], html)

    def test_history_lists_two_visits_for_same_patient(self):
        self.client.post(reverse("predict"), data=VALID_LOW_RISK)
        second_visit = dict(VALID_LOW_RISK)
        second_visit["oldpeak"] = 1.2
        self.client.post(reverse("predict"), data=second_visit)
        response = self.client.get(reverse("history"))
        self.assertEqual(len(response.context["records"]), 2)


class TestHomeView(TestCase):
    def test_home_stats_include_disagreement_count(self):
        self.client.post(reverse("predict"), data=VALID_HIGH_RISK)
        self.client.post(reverse("predict"), data=VALID_LOW_RISK)
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["total"], 2)
        self.assertIn("disagreements", response.context)
        self.assertEqual(response.context["disease"] + response.context["no_disease"], 2)

    def test_home_recent_log_shows_patient_id_but_never_name(self):
        self.client.post(reverse("predict"), data=VALID_HIGH_RISK)
        response = self.client.get(reverse("home"))

        recent_ctx = response.context["recent"][0]
        self.assertEqual(recent_ctx["patient_id"], VALID_HIGH_RISK["patient_id"])
        self.assertNotIn("patient_name", recent_ctx)

        html = response.content.decode()
        self.assertIn(VALID_HIGH_RISK["patient_id"], html)
        self.assertNotIn(VALID_HIGH_RISK["patient_name"], html)