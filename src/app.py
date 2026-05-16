import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
import io
from temporal_counterfactuals import TemporalCounterfactualPlanner
from uncertainty import risk_confidence
from risk_timeline import simulate_timeline

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "clean_data.csv"
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"

st.set_page_config(layout="wide")

# ==============================
# Load model & data
# ==============================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
planner = TemporalCounterfactualPlanner(MODEL_PATH)


df = pd.read_csv(DATA_PATH)

drop_cols = ['EarlyLabel', 'Patient_ID', 'Unnamed: 0']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])


# ==============================
# Header
# ==============================
st.title("Explainable Early Sepsis Prediction System")

st.write(
"Predict sepsis early and explain predictions using SHAP, counterfactuals, and interpretable rules."
)


# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("Patient Selection")

total_patients = len(X)
st.sidebar.info(f"Total Patients Available: {total_patients}")
# # Quick navigation buttons
# if st.sidebar.button("Show Highest Risk Patient"):
#     probs = model.predict_proba(X)[:, 1]
#     idx = int(np.argmax(probs))
#     sample = X.iloc[idx:idx+1]
#     st.sidebar.success(f"Showing highest risk patient (index {idx})")

# if st.sidebar.button("Random Patient"):
#     idx = np.random.randint(0, total_patients)
#     sample = X.iloc[idx:idx+1]
#     st.sidebar.success(f"Showing random patient (index {idx})")


# mode = st.sidebar.radio(
#     "Choose input",
#     ["Use existing patient", "Upload new patient"]
# )

# # --------------------------------------
# # Existing patient
# # --------------------------------------
# if mode == "Use existing patient":

#     idx = st.sidebar.number_input(
#         "Enter Patient Index",
#         min_value=0,
#         max_value=len(X)-1,
#         value=0,
#         step=1
#     )

#     sample = X.iloc[idx:idx+1]   # ✅ FIXED

# # --------------------------------------
# # Upload new patient
# # --------------------------------------
# else:
#     uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

#     if uploaded is not None:
#         new_patient = pd.read_csv(uploaded)

#         X = pd.concat([X, new_patient], ignore_index=True)
#         sample = new_patient

#         st.sidebar.success("New patient added!")
#         st.sidebar.info(f"Total Patients Now: {len(X)}")

#     else:
#         st.stop()
st.sidebar.header("Patient Selection")

total_patients = len(X)
st.sidebar.info(f"Total Patients Available: {total_patients}")

mode = st.sidebar.radio(
    "Choose input",
    ["Use existing patient", "Upload new patient"]
)

sample = None
idx = None

# ----------------------------
# EXISTING PATIENT
# ----------------------------
if mode == "Use existing patient":

    option = st.sidebar.radio(
        "Select method",
        ["Enter index", "Highest risk", "Random"]
    )

    if option == "Enter index":
        idx = st.sidebar.number_input(
            "Enter Patient Index",
            min_value=0,
            max_value=len(X)-1,
            value=0,
            step=1
        )

    elif option == "Highest risk":
        probs = model.predict_proba(X)[:, 1]
        idx = int(np.argmax(probs))

    elif option == "Random":
        idx = np.random.randint(0, len(X))

    sample = X.iloc[idx:idx+1]

# ----------------------------
# UPLOAD
# ----------------------------
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        new_patient = pd.read_csv(uploaded)
        sample = new_patient
        idx = "Uploaded Patient"

        st.sidebar.success("New patient added!")

    else:
        st.stop()

# ==========================================================
# Prediction
# ==========================================================
sample = sample.copy()   # ✅ ADD THIS LINE

prob = model.predict_proba(sample)[0][1]
risk = prob * 100
_, confidence = risk_confidence(model, sample)

st.write("Selected Patient:", idx)
st.write("Risk:", risk)
# ==========================================================
# SECTION 1 — Risk
# ==========================================================
st.header("🩺 Patient Risk Summary")

if risk < 30:
    st.success("🟢 LOW RISK: Patient is currently stable")
elif risk < 70:
    st.warning("🟠 MEDIUM RISK: Monitor patient closely")
else:
    st.error("🔴 HIGH RISK: Immediate attention required")

st.write(f"Predicted Sepsis Risk: {risk:.2f}%")

col1, col2 = st.columns(2)

if risk < 30:
    label = "LOW"
elif risk < 70:
    label = "MEDIUM"
else:
    label = "HIGH"

col1.metric("Predicted Risk", f"{risk:.2f}%")
col2.metric("Model Confidence", f"{confidence:.1f}%")
col2.write(f"### {label}")

probs = model.predict_proba(X)[:, 1] * 100

# ---------------- Top risky patients ----------------
st.subheader("Top 10 Highest Risk Patients")

top_idx = np.argsort(probs)[-10:][::-1]

top_table = pd.DataFrame({
    "Rank": range(1, 11),
    "Patient Index": top_idx,
    "Risk %": probs[top_idx]
})

st.dataframe(top_table)


# ---------------- Full ranking ----------------
st.subheader("Patient Risk Ranking (All Patients)")

ranking_df = pd.DataFrame({
    "Patient Index": range(len(probs)),
    "Risk %": probs
}).sort_values("Risk %", ascending=False)

st.dataframe(ranking_df, height=300)


# ==========================================================
# SECTION 2 — SHAP Explanation (WITH TEXT)
# ==========================================================
st.header("Why did the model predict this?")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)
values = shap_values[0]

importance = sorted(
    zip(sample.columns, values),
    key=lambda x: abs(x[1]),
    reverse=True
)

increase = [(f, v) for f, v in importance if v > 0][:3]
decrease = [(f, v) for f, v in importance if v < 0][:3]

st.subheader("Explanation in simple words")

def explain_feature(f):
    explanations = {
        "HR": "Heart rate level",
        "Temp": "Body temperature",
        "MAP": "Blood pressure stability",
        "SBP": "Systolic blood pressure",
        "Resp": "Breathing rate",
        "Lactate": "Oxygen delivery efficiency",
        "O2Sat": "Blood oxygen level",
        "ICULOS": "Length of ICU stay"
    }
    return explanations.get(f, f)


if len(increase) > 0:
    st.write("Factors increasing risk:")
    for f, v in increase:
        st.write(f"• {explain_feature(f)} is increasing risk")

if len(decrease) > 0:
    st.write("Factors decreasing risk:")
    for f, v in decrease:
        st.write(f"• {explain_feature(f)} is decreasing risk")

st.info("Red features push risk higher. Blue features push risk lower.")

# SHAP plot
fig = plt.figure(figsize=(6, 3))  # smaller size

shap.waterfall_plot(
    shap.Explanation(
        values=values,
        base_values=explainer.expected_value,
        data=sample.iloc[0],
        feature_names=sample.columns
    ),
    show=False
)

# # Center + prevent stretching
# col1, col2, col3 = st.columns([1,3,1])

# with col2:
st.pyplot(fig, use_container_width=False)


# ==========================================================
# SECTION 3 — Counterfactual
# ==========================================================
st.header("How to reduce risk?")

cf = sample.copy()

normal_values = {
    'HR': 75,
    'Temp': 36.8,
    'MAP': 85,
    'SBP': 120,
    'Resp': 16,
    'Lactate': 1.2,
    'O2Sat': 98
}

rows = []

current_prob = prob

for col, val in normal_values.items():

    if col not in cf.columns:
        continue

    original_val = cf.iloc[0][col]

    temp_cf = cf.copy()
    temp_cf[col] = val

    new_prob = model.predict_proba(temp_cf)[0][1]

    # only accept change if risk decreases
    if new_prob < current_prob:
        cf[col] = val
        rows.append([col, original_val, val])
        current_prob = new_prob


new_prob = current_prob


st.success(
    f"Predicted risk reduces from {risk:.2f}% → {new_prob * 100:.2f}% "
    "after stabilizing key vitals."
)

# ---------- SIDE-BY-SIDE COMPARISON ----------
colA, colB = st.columns(2)

with colA:
    st.subheader("Risk Before vs After Treatment")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(
        ["Before Treatment", "After Treatment"],
        [risk, new_prob * 100],
        marker='o'
    )
    ax.set_ylabel("Risk (%)")
    ax.set_title("Effect of Treatment on Risk")
    st.pyplot(fig)

with colB:
    st.subheader("Feature Comparison (Before vs After)")
    comparison_df = pd.DataFrame(
        rows,
        columns=["Vital Sign", "Before Treatment", "After Treatment"]
    )
    st.dataframe(comparison_df, height=260)


# ==========================================================
# NOVEL FEATURE — SEPSIS RISK TIMELINE
# ==========================================================
st.header("⏳ Future Sepsis Risk Prediction (Novel Feature)")

scenario = st.radio(
    "Select Scenario",
    ["No Treatment (Condition Worsens)", "With Treatment (Condition Improves)"],
    key="timeline_scenario"
)

treatment_flag = scenario == "With Treatment (Condition Improves)"

# ✅ THIS LINE MUST COME BEFORE PLOT
future_risks = simulate_timeline(
    model,
    sample.copy(),
    hours=6,
    treatment=treatment_flag
)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 2.5))  # smaller size

ax.plot(range(1,7), future_risks, marker='o')
ax.set_xlabel("Next 6 Hours")
ax.set_ylabel("Risk (%)")
ax.set_title("Predicted Risk Progression")

st.pyplot(fig, use_container_width=False)  # 👈 IMPORTANT


# =========================
# Critical Alert System
# =========================
threshold = 80

alert_triggered = False

for i, r in enumerate(future_risks):
    if r >= threshold:
        st.error(f"🚨 Risk expected to cross {threshold}% in {i+1} hours")
        alert_triggered = True
        break

if not alert_triggered:
    st.success("No critical risk threshold expected in next 6 hours.")

# =========================
# Explanation
# =========================
st.subheader("What does this mean?")

if treatment_flag:
    st.write(
        "With treatment, patient vitals improve over time, reducing sepsis risk."
    )
else:
    st.write(
        "Without treatment, patient condition worsens, leading to increasing sepsis risk."
    )


# ==========================================================
# ⭐ NOVEL SECTION — Actionable Intervention Plan
# ==========================================================
st.header("🧭 Suggested Treatment Plan")

plan = planner.plan(sample.iloc[0], horizon=3, target=0.10)

def treatment_reason(feature):
    reasons = {
        "MAP": "Improves blood circulation and organ perfusion",
        "SBP": "Stabilizes blood pressure",
        "HR": "Normalizes heart activity",
        "Temp": "Controls infection and fever",
        "Resp": "Improves breathing efficiency",
        "Lactate": "Indicates better oxygen delivery",
        "O2Sat": "Improves oxygen levels in blood"
    }
    return reasons.get(feature, "Improves patient stability")


if len(plan) == 0:
    st.info("No large improvements found, but vitals already near optimal.")

else:
    for s in plan:
        st.write(f"""
            **Step {s['step']}**

            🔧 Action: Adjust {s['feature']} from {s['old']:.2f} → {s['new']:.2f}  
            🧠 Reason: {treatment_reason(s['feature'])}  
            📉 Result: Risk becomes {(s['risk_after']*100):.2f}%
            """)
        

# ==========================================================
# SECTION 4 — Patient Specific Rules (BETTER EXPLANATION)
# ==========================================================
st.header("Clinical Interpretation")

triggered = []

if 'Temp' in sample and sample['Temp'].iloc[0] > 38:
    triggered.append("Fever detected (Temp > 38°C)")

if 'Lactate' in sample and sample['Lactate'].iloc[0] > 2:
    triggered.append("High lactate → poor tissue perfusion")

if 'MAP' in sample and sample['MAP'].iloc[0] < 65:
    triggered.append("Low blood pressure (MAP < 65)")

if 'pH' in sample and sample['pH'].iloc[0] < 7.38:
    triggered.append("Acidosis (low pH)")

if 'Resp' in sample and sample['Resp'].iloc[0] > 22:
    triggered.append("High respiratory rate")

if 'ICULOS' in sample and sample['ICULOS'].iloc[0] > 48:
    triggered.append("Prolonged ICU stay")


if len(triggered) > 0:
    st.write("Conditions contributing to higher sepsis risk:")
    for r in triggered:
        st.write("•", r)
else:
    st.success(
        "No major abnormal clinical conditions detected. "
        "Patient vitals are currently within normal range."
    )


st.subheader("General medical knowledge used by the model")

general_rules = [
    "Fever increases infection risk",
    "High lactate indicates shock",
    "Low blood pressure suggests organ failure risk",
    "Acidosis is common in severe infections",
    "Long ICU stay increases complications"
]

for r in general_rules:
    st.write("•", r)

# ==========================================================
# SECTION 5 — PDF EXPORT
# ==========================================================
st.header("Export Report")

def generate_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Sepsis Risk Clinical Report", styles["Heading1"]))
    elements.append(Spacer(1, 20))

    # Risk summary
    elements.append(Paragraph(f"Predicted Risk: {risk:.2f}%", styles["Normal"]))
    elements.append(Paragraph(f"Risk Level: {label}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Explanation
    elements.append(Paragraph("Top Risk Factors:", styles["Heading2"]))
    for f, _ in increase:
        elements.append(Paragraph(f"- {f} increased risk", styles["Normal"]))

    elements.append(Spacer(1, 20))

    # Counterfactual
    elements.append(Paragraph("Suggested Improvements:", styles["Heading2"]))
    for row in rows:
        elements.append(Paragraph(f"- {row[0]}: {row[1]} → {row[2]}", styles["Normal"]))

    elements.append(Spacer(1, 20))

    # Rules
    elements.append(Paragraph("Triggered Clinical Conditions:", styles["Heading2"]))
    if len(triggered):
        for r in triggered:
            elements.append(Paragraph(f"- {r}", styles["Normal"]))
    else:
        elements.append(Paragraph("Vitals within normal range", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

st.header("📌 Final Clinical Interpretation")

if risk < 30:
    st.success("""
Patient is stable with normal vital signs.
No strong indicators of sepsis are present.
Routine monitoring is sufficient.
""")

elif risk < 70:
    st.warning("""
Some abnormal vitals detected.
Patient should be closely monitored for early signs of sepsis.
""")

else:
    st.error("""
High-risk patient with multiple abnormal indicators.
Immediate clinical intervention is recommended.
""")

pdf_file = generate_pdf()

st.download_button(
    label="Download Patient Report (PDF)",
    data=pdf_file,
    file_name="sepsis_report.pdf",
    mime="application/pdf"
)
