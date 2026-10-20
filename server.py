import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict

warnings.filterwarnings("ignore")

app = FastAPI(
    title="HemoVision AI — Clinical Hemoglobinopathy Decision Support",
    description="Proof of Concept for HPLC AI Model with Built-in Explainability Suite",
    version="1.1.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Paths: Prioritize best_logistic_regression_model.joblib
LOGISTIC_PATH = os.path.join(BASE_DIR, "best_logistic_regression_model.joblib")
SVM_PATH = os.path.join(BASE_DIR, "best_svm_model.joblib")
if not os.path.exists(SVM_PATH):
    SVM_PATH = os.path.join(BASE_DIR, "best_svm_model (1).joblib")

if os.path.exists(LOGISTIC_PATH):
    MODEL_PATH = LOGISTIC_PATH
    MODEL_TYPE_NAME = "Logistic Regression Classifier"
else:
    MODEL_PATH = SVM_PATH
    MODEL_TYPE_NAME = "Support Vector Machine Classifier"

SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")

# Dataset Paths: Prioritize HPLC_data_cleaned (3).xlsx
DATASET_XLSX = os.path.join(BASE_DIR, "HPLC_data_cleaned (3).xlsx")
DATASET_CSV = os.path.join(BASE_DIR, "HPLC_data_cleaned (3).csv")

# 1. Load Model and Scaler
print(f"Loading AI model from {MODEL_PATH} ({MODEL_TYPE_NAME})...")
model = joblib.load(MODEL_PATH)
print(f"Loading Scaler from {SCALER_PATH}...")
scaler = joblib.load(SCALER_PATH)

FEATURES = list(scaler.feature_names_in_)
FEATURE_METADATA = {
    "Age_years": {
        "label": "Age",
        "unit": "Years",
        "min": 0.1,
        "max": 80.0,
        "step": 0.5,
        "default": 28.0,
        "normal_range": [0, 80],
        "desc": "Patient chronological age in years."
    },
    "RBC Count": {
        "label": "RBC Count",
        "unit": "10¹²/L",
        "min": 1.0,
        "max": 8.5,
        "step": 0.05,
        "default": 5.5,
        "normal_range": [4.0, 5.8],
        "desc": "Red Blood Cell count. Often elevated (>5.5) in Beta Thalassemia Trait despite low Hb."
    },
    "HCT": {
        "label": "Hematocrit (HCT)",
        "unit": "%",
        "min": 10.0,
        "max": 60.0,
        "step": 0.5,
        "default": 35.0,
        "normal_range": [36.0, 48.0],
        "desc": "Volume percentage of red blood cells in blood."
    },
    "Hemoglobin": {
        "label": "Hemoglobin (Hb)",
        "unit": "g/dL",
        "min": 2.0,
        "max": 19.0,
        "step": 0.1,
        "default": 11.2,
        "normal_range": [11.5, 16.5],
        "desc": "Total hemoglobin concentration."
    },
    "MCV": {
        "label": "MCV",
        "unit": "fL",
        "min": 40.0,
        "max": 125.0,
        "step": 0.5,
        "default": 64.0,
        "normal_range": [80.0, 100.0],
        "desc": "Mean Corpuscular Volume. <80 fL indicates microcytosis, key to both IDA and Thalassemia."
    },
    "MCH": {
        "label": "MCH",
        "unit": "pg",
        "min": 10.0,
        "max": 42.0,
        "step": 0.2,
        "default": 20.0,
        "normal_range": [26.0, 34.0],
        "desc": "Mean Corpuscular Hemoglobin. Average mass of Hb per red blood cell."
    },
    "MCHC": {
        "label": "MCHC",
        "unit": "g/dL",
        "min": 18.0,
        "max": 40.0,
        "step": 0.2,
        "default": 31.5,
        "normal_range": [31.5, 36.0],
        "desc": "Mean Corpuscular Hemoglobin Concentration. Markedly depressed in IDA."
    },
    "RDW-CV": {
        "label": "RDW-CV",
        "unit": "%",
        "min": 10.0,
        "max": 38.0,
        "step": 0.2,
        "default": 17.5,
        "normal_range": [11.5, 15.0],
        "desc": "Red Cell Distribution Width. Anisocytosis marker; severely high (>18%) in IDA, lower in BTT."
    },
    "Mentzer Index": {
        "label": "Mentzer Index",
        "unit": "Ratio",
        "min": 5.0,
        "max": 65.0,
        "step": 0.1,
        "default": 11.6,
        "normal_range": [11.0, 15.0],
        "desc": "MCV / RBC. Classic clinical cutoff: <13 suggests BTT, >13 suggests IDA."
    }
}

# 2. Process Dataset for Cohort Analysis
def parse_age(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    if 'month' in x:
        return round(float(x.split('_')[0]) / 12.0, 2)
    elif 'day' in x:
        return round(float(x.split('_')[0]) / 365.25, 3)
    elif 'yr' in x:
        return float(x.split('_')[0])
    try:
        return float(x)
    except:
        return np.nan

print("Processing dataset for cohort analysis...")
if os.path.exists(DATASET_XLSX):
    print(f"Loading dataset from Excel: {DATASET_XLSX}...")
    raw_df = pd.read_excel(DATASET_XLSX)
elif os.path.exists(DATASET_CSV):
    print(f"Loading dataset from CSV: {DATASET_CSV}...")
    raw_df = pd.read_csv(DATASET_CSV)
else:
    raise FileNotFoundError("No HPLC dataset file found!")

raw_df["Age_years"] = raw_df["Age"].apply(parse_age)

# Impute missing features with medians
for col in FEATURES:
    if col in raw_df.columns:
        med = float(raw_df[col].median())
        raw_df[col] = raw_df[col].fillna(med)

raw_df["patient_id"] = [f"PT-{i+1:04d}" for i in range(len(raw_df))]

# Standardize diagnosis labels
def normalize_diagnosis(val):
    s = str(val).strip().lower()
    if "iron deficiency" in s or s == "ida":
        return "Iron Deficiency Anemia"
    elif "beta thalassemia" in s or "heterozygous" in s:
        return "Beta Thalassemia Trait"
    elif "combined" in s:
        return "Combined IDA & Beta Thalassemia Trait"
    elif "normal" in s:
        return "Normal"
    return str(val).strip()

raw_df["standard_diagnosis"] = raw_df["HPLC Diagnosis"].apply(normalize_diagnosis)

# Primary diagnostic scope (IDA vs BTT)
btt_ida_df = raw_df[raw_df["standard_diagnosis"].isin(["Iron Deficiency Anemia", "Beta Thalassemia Trait"])].copy()
btt_ida_df["y_true"] = btt_ida_df["standard_diagnosis"].apply(lambda x: 0 if x == "Iron Deficiency Anemia" else 1)

X_btt_ida = btt_ida_df[FEATURES].values
X_btt_ida_scaled = scaler.transform(X_btt_ida)
btt_ida_df["svm_pred"] = model.predict(X_btt_ida_scaled)
btt_ida_df["svm_prob_btt"] = model.predict_proba(X_btt_ida_scaled)[:, 1]
btt_ida_df["svm_prob_ida"] = model.predict_proba(X_btt_ida_scaled)[:, 0]

# 3. Setup SHAP Explainer
print("Initializing SHAP background kernel explainer...")
bg_kmeans = shap.kmeans(X_btt_ida_scaled, 10)
shap_explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], bg_kmeans)
expected_val = float(np.ravel(shap_explainer.expected_value)[0])

# Precompute Global Feature Importance
shap_cohort_sample = X_btt_ida_scaled[:50]
shap_vals_cohort = shap_explainer.shap_values(shap_cohort_sample, nsamples=80, silent=True)
if isinstance(shap_vals_cohort, list):
    shap_vals_cohort = shap_vals_cohort[0]
if hasattr(shap_vals_cohort, 'values'):
    shap_vals_cohort = shap_vals_cohort.values
shap_vals_cohort = np.array(shap_vals_cohort)
global_importance = np.abs(shap_vals_cohort).mean(axis=0)
GLOBAL_FEATURE_IMPORTANCE = [
    {
        "feature": f,
        "importance": float(imp),
        "label": FEATURE_METADATA[f]["label"],
        "unit": FEATURE_METADATA[f]["unit"]
    }
    for f, imp in sorted(zip(FEATURES, global_importance), key=lambda x: x[1], reverse=True)
]

# 4. Extract Decision Boundary Exemplars
# For Logistic Regression, boundary cases closest to decision margin z = 0 serve as the anchor prototypes
if hasattr(model, 'support_vectors_'):
    sv_scaled = model.support_vectors_
    sv_unscaled = scaler.inverse_transform(sv_scaled)
    n_sv_ida = int(model.n_support_[0])
    n_sv_btt = int(model.n_support_[1])
else:
    decisions = model.decision_function(X_btt_ida_scaled)
    btt_ida_df["abs_decision"] = np.abs(decisions)
    ida_cases = btt_ida_df[btt_ida_df["y_true"] == 0].sort_values("abs_decision").head(20)
    btt_cases = btt_ida_df[btt_ida_df["y_true"] == 1].sort_values("abs_decision").head(18)
    exemplar_df = pd.concat([ida_cases, btt_cases])
    sv_unscaled = exemplar_df[FEATURES].values
    sv_scaled = scaler.transform(sv_unscaled)
    n_sv_ida = len(ida_cases)
    n_sv_btt = len(btt_cases)

SUPPORT_VECTORS_DATA = []
for i in range(len(sv_scaled)):
    is_btt = i >= n_sv_ida
    label = "Beta Thalassemia Trait" if is_btt else "Iron Deficiency Anemia"
    code = 1 if is_btt else 0
    row_dict = {
        "sv_id": f"EX-{i+1:02d}",
        "class_label": label,
        "class_code": code,
        "features": {f: float(round(sv_unscaled[i, idx], 2)) for idx, f in enumerate(FEATURES)},
        "scaled_vector": sv_scaled[i].tolist()
    }
    SUPPORT_VECTORS_DATA.append(row_dict)

# 5. Pydantic Models
class PatientInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    Age_years: float = Field(28.0, ge=0.0, le=100.0)
    RBC_Count: float = Field(5.5, ge=0.5, le=10.0, alias="RBC Count")
    HCT: float = Field(35.0, ge=5.0, le=70.0)
    Hemoglobin: float = Field(11.2, ge=1.0, le=25.0)
    MCV: float = Field(64.0, ge=30.0, le=130.0)
    MCH: float = Field(20.0, ge=5.0, le=50.0)
    MCHC: float = Field(31.5, ge=15.0, le=45.0)
    RDW_CV: float = Field(17.5, ge=8.0, le=45.0, alias="RDW-CV")
    Mentzer_Index: Optional[float] = Field(None, alias="Mentzer Index")

# Helper: Clinical Insights Generator
def generate_clinical_narrative(patient_data: Dict[str, float], shap_dict: Dict[str, float], prob_btt: float) -> List[str]:
    insights = []
    
    # RBC insights
    rbc = patient_data["RBC Count"]
    if rbc >= 5.5:
        insights.append(f"High Red Blood Cell Count ({rbc:.2f} 10¹²/L) reflects microcytic erythrocytosis, a hallmark of Beta Thalassemia Trait.")
    elif rbc < 4.2:
        insights.append(f"Subnormal RBC Count ({rbc:.2f} 10¹²/L) is characteristic of Iron Deficiency Anemia impaired erythropoiesis.")

    # RDW-CV insights
    rdw = patient_data["RDW-CV"]
    if rdw >= 18.0:
        insights.append(f"Markedly elevated RDW-CV ({rdw:.1f}%) demonstrates severe red cell size inequality (anisocytosis), heavily shifting diagnosis toward IDA.")
    elif rdw < 16.0:
        insights.append(f"Relatively homogeneous RDW-CV ({rdw:.1f}%) aligns with the uniform microcytosis typical of heterozygous thalassemia.")

    # Mentzer Index insights
    mentzer = patient_data["Mentzer Index"]
    if mentzer < 13.0:
        insights.append(f"Mentzer Index is {mentzer:.1f} (<13.0), which traditionally points to Beta Thalassemia Trait.")
    else:
        insights.append(f"Mentzer Index is {mentzer:.1f} (≥13.0), which traditionally points to Iron Deficiency Anemia.")

    # Hemoglobin & MCV
    mcv = patient_data["MCV"]
    if mcv < 75.0:
        insights.append(f"MCV of {mcv:.1f} fL confirms pronounced microcytosis.")

    return insights

# API Endpoints
@app.get("/api/info")
def get_model_info():
    is_logistic = hasattr(model, 'coef_')
    coef_dict = {}
    if is_logistic:
        for idx, f in enumerate(FEATURES):
            coef_dict[f] = round(float(model.coef_[0, idx]), 4)

    return {
        "model_name": "Logistic Regression Classifier (L2 Regularized)" if is_logistic else "RBF Support Vector Machine Classifier",
        "framework": "scikit-learn (LogisticRegression)" if is_logistic else "scikit-learn (SVC)",
        "model_family": "logistic_regression" if is_logistic else "svm",
        "hyperparameters": {
            "penalty": getattr(model, "penalty", "l2"),
            "C": float(getattr(model, "C", 1.0)),
            "solver": getattr(model, "solver", "lbfgs"),
            "random_state": getattr(model, "random_state", 42)
        },
        "coefficients": coef_dict,
        "intercept": float(model.intercept_[0]) if hasattr(model, 'intercept_') else 0.0,
        "support_vectors_count": int(len(sv_scaled)),
        "support_vectors_breakdown": {
            "Iron Deficiency Anemia (Class 0)": int(n_sv_ida),
            "Beta Thalassemia Trait (Class 1)": int(n_sv_btt)
        },
        "features": FEATURE_METADATA,
        "feature_order": FEATURES,
        "classes": {
            0: "Iron Deficiency Anemia (IDA)",
            1: "Beta Thalassemia Trait / Heterozygous (BTT)"
        },
        "accuracy_reported": 85.3,
        "sensitivity_btt": 99.6,
        "expected_shap_baseline": expected_val,
        "global_importance": GLOBAL_FEATURE_IMPORTANCE,
        "total_cohort_size": int(len(raw_df)),
        "ida_btt_cohort_size": int(len(btt_ida_df))
    }

@app.post("/api/predict")
def predict_patient(payload: PatientInput):
    data = payload.model_dump(by_alias=True)
    
    # Auto-compute Mentzer Index if not provided or zero
    if data["Mentzer Index"] is None or data["Mentzer Index"] <= 0:
        if data["RBC Count"] > 0:
            data["Mentzer Index"] = round(data["MCV"] / data["RBC Count"], 2)
        else:
            data["Mentzer Index"] = 13.0

    raw_values = np.array([[data[f] for f in FEATURES]], dtype=float)
    raw_df_pt = pd.DataFrame(raw_values, columns=FEATURES)
    scaled_vector = scaler.transform(raw_df_pt)

    # Predictions
    prob = model.predict_proba(scaled_vector)[0]
    prob_ida = float(prob[0])
    prob_btt = float(prob[1])
    decision_margin = float(model.decision_function(scaled_vector)[0])
    predicted_class = 1 if prob_btt >= 0.5 else 0
    diagnosis = "Beta Thalassemia Trait / Heterozygous" if predicted_class == 1 else "Iron Deficiency Anemia"
    confidence = float(max(prob_ida, prob_btt) * 100)

    # Local SHAP Attributions
    shap_vals = shap_explainer.shap_values(scaled_vector, nsamples=100, silent=True)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    if hasattr(shap_vals, 'values'):
        shap_vals = shap_vals.values
    shap_vals = np.array(shap_vals)
    if shap_vals.ndim > 1:
        shap_vals = shap_vals[0]

    waterfall = []
    shap_dict = {}
    for idx, f in enumerate(FEATURES):
        val = float(shap_vals[idx])
        shap_dict[f] = val
        waterfall.append({
            "feature": f,
            "label": FEATURE_METADATA[f]["label"],
            "unit": FEATURE_METADATA[f]["unit"],
            "patient_value": float(round(data[f], 2)),
            "attribution": float(round(val, 4)),
            "attribution_pct": float(round(val * 100, 2)),
            "direction": "btt" if val > 0 else "ida"
        })
    
    waterfall.sort(key=lambda x: abs(x["attribution"]), reverse=True)

    # Nearest Boundary Exemplars (Case-Based Reasoning)
    distances = np.linalg.norm(sv_scaled - scaled_vector, axis=1)
    nearest_idx = np.argsort(distances)[:3]
    nearest_support_vectors = []
    for idx in nearest_idx:
        sv_item = SUPPORT_VECTORS_DATA[idx]
        dist = float(distances[idx])
        similarity = float(round(100.0 / (1.0 + dist), 1))
        nearest_support_vectors.append({
            "sv_id": sv_item["sv_id"],
            "diagnosis": sv_item["class_label"],
            "class_code": sv_item["class_code"],
            "distance": round(dist, 3),
            "similarity_pct": similarity,
            "features": sv_item["features"]
        })

    # Mentzer Index Comparison & Discordance Check
    mentzer_val = data["Mentzer Index"]
    mentzer_suggests = "Beta Thalassemia Trait" if mentzer_val < 13.0 else "Iron Deficiency Anemia"
    svm_suggests = diagnosis
    concordance = "CONCORDANT" if (mentzer_val < 13.0 and predicted_class == 1) or (mentzer_val >= 13.0 and predicted_class == 0) else "DISCORDANT"
    
    mentzer_analysis = {
        "mentzer_value": mentzer_val,
        "rule_threshold": 13.0,
        "mentzer_verdict": mentzer_suggests,
        "svm_verdict": svm_suggests,
        "status": concordance,
        "explanation": (
            "Both the traditional Mentzer rule (<13) and AI decision boundary agree."
            if concordance == "CONCORDANT"
            else (
                f"Discordance detected! Traditional Mentzer Index ({mentzer_val:.1f}) suggests {mentzer_suggests}, "
                f"but the multi-marker AI model captures joint blood indices (including RDW-CV & RBC Count) "
                f"and concludes {svm_suggests} with {confidence:.1f}% confidence."
            )
        )
    }

    # Counterfactual "What-If" Minimal Perturbations
    rbc_deltas = [-1.0, -0.5, +0.5, +1.0]
    rbc_impact = []
    rbc_idx = FEATURES.index("RBC Count")
    m_idx = FEATURES.index("Mentzer Index")
    for d in rbc_deltas:
        curr_unscaled = data["RBC Count"] + d
        if 1.0 <= curr_unscaled <= 8.5:
            temp_raw = np.copy(raw_values)
            temp_raw[0, rbc_idx] = curr_unscaled
            temp_raw[0, m_idx] = data["MCV"] / curr_unscaled
            temp_df = pd.DataFrame(temp_raw, columns=FEATURES)
            temp_scaled = scaler.transform(temp_df)
            p_btt = float(model.predict_proba(temp_scaled)[0][1])
            rbc_impact.append({"delta": d, "rbc": round(curr_unscaled, 2), "new_btt_prob": round(p_btt * 100, 1)})

    rdw_deltas = [-3.0, -1.5, +1.5, +3.0]
    rdw_impact = []
    rdw_idx = FEATURES.index("RDW-CV")
    for d in rdw_deltas:
        curr_unscaled = data["RDW-CV"] + d
        if 10.0 <= curr_unscaled <= 35.0:
            temp_raw = np.copy(raw_values)
            temp_raw[0, rdw_idx] = curr_unscaled
            temp_df = pd.DataFrame(temp_raw, columns=FEATURES)
            temp_scaled = scaler.transform(temp_df)
            p_btt = float(model.predict_proba(temp_scaled)[0][1])
            rdw_impact.append({"delta": d, "rdw": round(curr_unscaled, 2), "new_btt_prob": round(p_btt * 100, 1)})

    clinical_narratives = generate_clinical_narrative(data, shap_dict, prob_btt)

    return {
        "diagnosis": diagnosis,
        "predicted_class": predicted_class,
        "probabilities": {
            "btt": round(prob_btt * 100, 2),
            "ida": round(prob_ida * 100, 2)
        },
        "confidence": round(confidence, 1),
        "decision_margin": round(decision_margin, 3),
        "margin_zone": "Strong Confidence" if abs(decision_margin) > 0.8 else ("Moderate" if abs(decision_margin) > 0.3 else "Boundary / Borderline"),
        "shap_analysis": {
            "baseline_prob": round(expected_val * 100, 2),
            "final_prob": round(prob_btt * 100, 2),
            "waterfall": waterfall
        },
        "mentzer_comparison": mentzer_analysis,
        "nearest_support_vectors": nearest_support_vectors,
        "sensitivity": {
            "rbc_sweep": rbc_impact,
            "rdw_sweep": rdw_impact
        },
        "clinical_narratives": clinical_narratives,
        "evaluated_features": data
    }

@app.get("/api/decision-boundary")
def get_decision_boundary(
    x_feature: str = Query("RBC Count"),
    y_feature: str = Query("MCV"),
    age: float = Query(28.0),
    hct: float = Query(35.0),
    hb: float = Query(11.2),
    mch: float = Query(20.0),
    mchc: float = Query(31.5),
    rdw: float = Query(17.5),
    grid_size: int = Query(26)
):
    if x_feature not in FEATURES or y_feature not in FEATURES:
        raise HTTPException(status_code=400, detail="Invalid feature name for 2D boundary")

    x_meta = FEATURE_METADATA[x_feature]
    y_meta = FEATURE_METADATA[y_feature]

    x_vals = np.linspace(x_meta["min"], x_meta["max"], grid_size)
    y_vals = np.linspace(y_meta["min"], y_meta["max"], grid_size)
    xx, yy = np.meshgrid(x_vals, y_vals)

    base_dict = {
        "Age_years": age,
        "RBC Count": 5.5,
        "HCT": hct,
        "Hemoglobin": hb,
        "MCV": 64.0,
        "MCH": mch,
        "MCHC": mchc,
        "RDW-CV": rdw,
        "Mentzer Index": 12.0
    }

    num_pts = xx.size
    grid_matrix = np.zeros((num_pts, len(FEATURES)))
    
    for idx, f in enumerate(FEATURES):
        grid_matrix[:, idx] = base_dict[f]

    x_idx = FEATURES.index(x_feature)
    y_idx = FEATURES.index(y_feature)
    grid_matrix[:, x_idx] = xx.ravel()
    grid_matrix[:, y_idx] = yy.ravel()

    m_idx = FEATURES.index("Mentzer Index")
    rbc_pos = FEATURES.index("RBC Count")
    mcv_pos = FEATURES.index("MCV")
    safe_rbc = np.clip(grid_matrix[:, rbc_pos], 0.1, 15.0)
    grid_matrix[:, m_idx] = grid_matrix[:, mcv_pos] / safe_rbc

    grid_df = pd.DataFrame(grid_matrix, columns=FEATURES)
    scaled_grid = scaler.transform(grid_df)
    probs_btt = model.predict_proba(scaled_grid)[:, 1].reshape(xx.shape)
    decisions = model.decision_function(scaled_grid).reshape(xx.shape)

    cohort_pts = []
    # Take a representative sample of 120 cohort points for smooth canvas rendering
    sample_cohort = btt_ida_df.sample(n=min(120, len(btt_ida_df)), random_state=42)
    for _, row in sample_cohort.iterrows():
        cohort_pts.append({
            "id": row["patient_id"],
            "x": float(row[x_feature]),
            "y": float(row[y_feature]),
            "true_diagnosis": row["standard_diagnosis"],
            "pred": int(row["svm_pred"])
        })

    sv_pts = []
    for sv in SUPPORT_VECTORS_DATA:
        sv_pts.append({
            "id": sv["sv_id"],
            "x": sv["features"][x_feature],
            "y": sv["features"][y_feature],
            "class_label": sv["class_label"],
            "class_code": sv["class_code"]
        })

    return {
        "x_feature": x_feature,
        "y_feature": y_feature,
        "x_range": [float(x_meta["min"]), float(x_meta["max"])],
        "y_range": [float(y_meta["min"]), float(y_meta["max"])],
        "grid_size": grid_size,
        "x_vals": x_vals.tolist(),
        "y_vals": y_vals.tolist(),
        "probs_btt": probs_btt.round(3).tolist(),
        "decisions": decisions.round(3).tolist(),
        "cohort_points": cohort_pts,
        "support_vectors": sv_pts
    }

@app.get("/api/cohort")
def get_cohort(filter_type: str = Query("ida_btt")):
    if filter_type == "all":
        target_df = raw_df
    else:
        target_df = btt_ida_df

    # Limit to top 250 rows for snappy response if dataset is large, or provide full cohort
    display_df = target_df.head(300)
    records = []
    for _, row in display_df.iterrows():
        rec = {
            "id": str(row.get("patient_id", "")),
            "age": str(row.get("Age", "")),
            "age_years": float(row.get("Age_years", 0.0)),
            "rbc": float(row.get("RBC Count", 0.0)),
            "hct": float(row.get("HCT", 0.0)),
            "hb": float(row.get("Hemoglobin", 0.0)),
            "mcv": float(row.get("MCV", 0.0)),
            "mch": float(row.get("MCH", 0.0)),
            "mchc": float(row.get("MCHC", 0.0)),
            "rdw": float(row.get("RDW-CV", 0.0)),
            "mentzer": float(row.get("Mentzer Index", 0.0)),
            "hplc_diagnosis": str(row.get("standard_diagnosis", row.get("HPLC Diagnosis", ""))),
            "svm_pred": int(row.get("svm_pred", 0)) if "svm_pred" in row and pd.notna(row["svm_pred"]) else None,
            "svm_prob_btt": float(round(row.get("svm_prob_btt", 0.0) * 100, 1)) if "svm_prob_btt" in row and pd.notna(row["svm_prob_btt"]) else None
        }
        records.append(rec)
    return {
        "count": len(target_df),
        "displayed_count": len(records),
        "records": records
    }

@app.get("/api/presets")
def get_presets():
    return {
        "classic_btt": {
            "id": "classic_btt",
            "name": "Classic Beta Thalassemia Trait",
            "description": "Marked microcytosis with compensatory erythrocytosis (high RBC 6.54, MCV 58.4, Mentzer 8.9).",
            "values": {
                "Age_years": 42.0,
                "RBC Count": 6.54,
                "HCT": 38.2,
                "Hemoglobin": 12.3,
                "MCV": 58.4,
                "MCH": 18.8,
                "MCHC": 32.2,
                "RDW-CV": 18.9,
                "Mentzer Index": 8.9
            }
        },
        "severe_ida": {
            "id": "severe_ida",
            "name": "Severe Iron Deficiency Anemia",
            "description": "Depressed RBC count, hypochromia, severe anisocytosis with high RDW-CV (25.2%).",
            "values": {
                "Age_years": 25.0,
                "RBC Count": 4.31,
                "HCT": 19.3,
                "Hemoglobin": 5.0,
                "MCV": 44.8,
                "MCH": 11.6,
                "MCHC": 25.9,
                "RDW-CV": 25.7,
                "Mentzer Index": 10.4
            }
        },
        "borderline_discordant": {
            "id": "borderline_discordant",
            "name": "Diagnostic Conflict: Mentzer vs AI",
            "description": "Mentzer index > 13 suggests IDA, but patient exhibits erythrocytosis and low RDW, leading model to identify Beta Thalassemia Trait.",
            "values": {
                "Age_years": 24.0,
                "RBC Count": 4.43,
                "HCT": 28.0,
                "Hemoglobin": 7.7,
                "MCV": 63.2,
                "MCH": 17.4,
                "MCHC": 27.5,
                "RDW-CV": 20.3,
                "Mentzer Index": 14.3
            }
        },
        "pediatric_microcytic": {
            "id": "pediatric_microcytic",
            "name": "Pediatric Microcytic Presentation",
            "description": "11-month-old infant with borderline blood indices needing definitive screening.",
            "values": {
                "Age_years": 0.92,
                "RBC Count": 4.57,
                "HCT": 22.6,
                "Hemoglobin": 5.4,
                "MCV": 49.5,
                "MCH": 11.8,
                "MCHC": 23.9,
                "RDW-CV": 25.2,
                "Mentzer Index": 10.8
            }
        }
    }

# Mount static folder
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/")
def serve_root():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
