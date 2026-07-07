from django.contrib import admin
from .models import Patient, Assessment


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ("id", "mrn", "name", "created_at", "assessment_count")
    search_fields = ("mrn", "name")
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)

    @admin.display(description="# Assessments")
    def assessment_count(self, obj):
        return obj.assessments.count()


@admin.register(Assessment)
class AssessmentAdmin(admin.ModelAdmin):
    list_display = (
        "id", "created_at", "patient_mrn", "age", "sex",
        "id3_pred", "c45_pred", "rf_pred", "models_agree", "is_unanimous", "verdict_label",
    )
    list_filter = ("sex", "chest_pain_type", "st_slope", "id3_pred", "c45_pred", "rf_pred")
    search_fields = ("patient__mrn", "patient__name")
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)

    @admin.display(description="Patient MRN")
    def patient_mrn(self, obj):
        return obj.patient.mrn

    @admin.display(boolean=True, description="Agree?")
    def models_agree(self, obj):
        return obj.models_agree

    @admin.display(description="Unanimous?")
    def is_unanimous(self, obj):
        if obj.unanimous is None:
            return "n/a (no RF vote)"
        return "Yes" if obj.unanimous else "No (2-1 split)"

    @admin.display(description="Verdict")
    def verdict_label(self, obj):
        return "Disease" if obj.verdict == 1 else "Normal"