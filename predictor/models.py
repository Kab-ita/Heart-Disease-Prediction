from django.db import models


class Patient(models.Model):
    mrn = models.CharField(max_length=50, unique=True, verbose_name="MRN (Patient ID)")
    name = models.CharField(max_length=150)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def save(self, *args, **kwargs):
        self.mrn = self.mrn.strip().upper()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.name} ({self.mrn})"


class Assessment(models.Model): 
    patient = models.ForeignKey(Patient, on_delete=models.PROTECT, related_name="assessments")
 
    age = models.IntegerField()
    sex = models.CharField(max_length=5)
    chest_pain_type = models.CharField(max_length=10)
    resting_bp = models.IntegerField()
    cholesterol = models.IntegerField()
    fasting_bs = models.IntegerField()
    resting_ecg = models.CharField(max_length=15)
    max_hr = models.IntegerField()
    exercise_angina = models.CharField(max_length=5)
    oldpeak = models.FloatField()
    st_slope = models.CharField(max_length=10) 

    id3_pred = models.IntegerField()
    c45_pred = models.IntegerField()
    rf_pred = models.IntegerField(null=True, blank=True)
    rf_confidence = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Assessment #{self.pk} for {self.patient.mrn} ({self.created_at:%Y-%m-%d %H:%M})"

    @property
    def votes(self):
        v = [self.id3_pred, self.c45_pred]
        if self.rf_pred is not None:
            v.append(self.rf_pred)
        return v

    @property
    def models_agree(self): 
        return len(set(self.votes)) == 1

    @property
    def unanimous(self): 
        if self.rf_pred is None:
            return None
        return self.id3_pred == self.c45_pred == self.rf_pred

    @property
    def verdict(self): 
        votes = self.votes
        if len(votes) == 3:
            return 1 if votes.count(1) > votes.count(0) else 0
        return self.id3_pred if self.id3_pred == self.c45_pred else self.c45_pred