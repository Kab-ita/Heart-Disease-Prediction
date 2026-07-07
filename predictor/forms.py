from django import forms

from .models import Patient


class PatientForm(forms.Form):
    patient_name = forms.CharField(label="Patient Name", max_length=150)
    patient_id = forms.CharField(label="Patient ID (MRN)", max_length=50)

    age = forms.IntegerField(label="Age (Years)", initial=50, min_value=1, max_value=120)
    sex = forms.ChoiceField(choices=[("M", "Male"), ("F", "Female")])
    chest_pain_type = forms.ChoiceField(choices=[("ATA", "ATA"), ("NAP", "NAP"), ("ASY", "ASY"), ("TA", "TA")])
    resting_bp = forms.IntegerField(label="Resting Blood Pressure (mm Hg)", initial=120, min_value=1, max_value=300)
    cholesterol = forms.IntegerField(label="Serum Cholesterol (mg/dl)", initial=200, min_value=1, max_value=700)
    fasting_bs = forms.TypedChoiceField(
        label="Fasting Blood Sugar > 120 mg/dl",
        choices=[(1, "Yes"), (0, "No")],
        coerce=int,
    )
    resting_ecg = forms.ChoiceField(choices=[("Normal", "Normal"), ("ST", "ST"), ("LVH", "LVH")])
    max_hr = forms.IntegerField(label="Max Heart Rate Achieved", initial=150, min_value=40, max_value=250)
    exercise_angina = forms.ChoiceField(label="Exercise Induced Angina", choices=[("Y", "Yes"), ("N", "No")])
    oldpeak = forms.FloatField(label="ST Depression (Oldpeak)", initial=0.0, min_value=-3.0, max_value=10.0)
    st_slope = forms.ChoiceField(choices=[("Up", "Up"), ("Flat", "Flat"), ("Down", "Down")])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update({"class": "form-control form-control-sm mb-2"})

    def clean_patient_id(self):
        mrn = self.cleaned_data["patient_id"].strip().upper()
        if not mrn:
            raise forms.ValidationError("Patient ID cannot be blank.")
        return mrn

    def clean_patient_name(self):
        name = self.cleaned_data["patient_name"].strip()
        if not name:
            raise forms.ValidationError("Patient name cannot be blank.")
        return name

    def clean(self):
        cleaned = super().clean()
        mrn = cleaned.get("patient_id")
        name = cleaned.get("patient_name")
        if mrn and name:
            existing = Patient.objects.filter(mrn=mrn).first()
            if existing and existing.name.strip().lower() != name.strip().lower():
                self.add_error(
                    "patient_name",
                    f'Patient ID "{mrn}" is already on file under a different name '
                    f'("{existing.name}"). Double-check the ID before continuing.'
                )
        return cleaned