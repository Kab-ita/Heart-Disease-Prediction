def bin_age(age):
    if age < 35: return "Young"
    elif age <= 55: return "Middle-Aged"
    return "Senior"

def bin_resting_bp(bp):
    if bp < 120: return "Normal"
    elif bp < 140: return "Pre-Hypertension"
    return "Hypertension"

def bin_cholesterol(chol):
    if chol < 200: return "Desirable"
    elif chol < 240: return "Borderline"
    return "High"

def bin_max_hr(hr):
    if hr < 100: return "Low"
    elif hr < 150: return "Moderate"
    return "High"

def bin_oldpeak(op):
    if op <= 0: return "None"
    elif op <= 2: return "Mild"
    return "Severe"

def discretize_row(row_dict):
    return {
        "Age": bin_age(float(row_dict["Age"])),
        "Sex": str(row_dict["Sex"]),
        "ChestPainType": str(row_dict["ChestPainType"]),
        "RestingBP": bin_resting_bp(float(row_dict["RestingBP"])),
        "Cholesterol": bin_cholesterol(float(row_dict["Cholesterol"])),
        "FastingBS": "Yes" if int(row_dict["FastingBS"]) == 1 else "No",
        "RestingECG": str(row_dict["RestingECG"]),
        "MaxHR": bin_max_hr(float(row_dict["MaxHR"])),
        "ExerciseAngina": str(row_dict["ExerciseAngina"]),
        "Oldpeak": bin_oldpeak(float(row_dict["Oldpeak"])),
        "ST_Slope": str(row_dict["ST_Slope"])
    }