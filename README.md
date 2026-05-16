# 🩺 Explainable Early Sepsis Prediction System

## 📌 Overview

This project presents an **Explainable AI (XAI) based Early Sepsis Prediction System** designed for ICU environments. The system predicts the likelihood of sepsis using patient vital signs and laboratory measurements while also providing interpretable explanations and actionable insights for clinicians.

Unlike traditional black-box machine learning systems, this framework focuses on:

- Early risk prediction
- Explainability
- Risk reduction simulation
- Clinical interpretation
- Future risk forecasting

The project integrates machine learning, Explainable AI, and interactive visualization into a unified clinical decision-support dashboard.

---

# 🚀 Features

## ✅ Sepsis Risk Prediction
Predicts the probability of sepsis using ICU patient data.

## ✅ SHAP Explainability
Explains **why** the model predicted a particular risk score using SHAP feature attribution.

## ✅ Counterfactual Risk Reduction
Shows how modifying important physiological parameters can reduce predicted risk.

## ✅ Clinical Rule Interpretation
Provides human-readable medical interpretations such as:
- Fever detection
- Hypotension
- Acidosis
- Elevated lactate

## ✅ Future Risk Forecasting (Novel Feature)
Simulates how sepsis risk may evolve over the next few hours under:
- worsening conditions
- improving treatment scenarios

## ✅ Critical Alert System (Novel Feature)
Generates warnings if projected risk crosses a dangerous threshold.

## ✅ PDF Report Generation
Allows exporting patient reports for documentation and clinical review.

---

# 🧠 Novel Contributions

This project extends beyond traditional prediction systems by introducing:

### 🔹 Temporal Risk Forecasting
Predicts future sepsis progression trends over time.

### 🔹 Treatment Simulation
Compares risk before and after simulated intervention.

### 🔹 Confidence-Aware Predictions
Quantifies reliability of predictions.

### 🔹 Actionable Explainability
Not only explains risk but also suggests how to reduce it.

---

# 📂 Dataset

The project uses ICU patient records from the **PhysioNet Sepsis Dataset**.

### Features include:
- Heart Rate (HR)
- Temperature (Temp)
- Mean Arterial Pressure (MAP)
- Oxygen Saturation (O2Sat)
- Respiratory Rate (Resp)
- Lactate
- Blood Pressure
- pH
- ICU Length of Stay

---

# 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core development |
| XGBoost | Sepsis prediction model |
| SHAP | Explainable AI |
| Pandas | Data preprocessing |
| Scikit-learn | ML utilities |
| Streamlit | Interactive dashboard |
| Matplotlib | Visualization |
| ReportLab | PDF generation |

---

# 📁 Project Structure

```bash
sepsis-xai/
│
├── data/
│   └── clean_data.csv
│
├── models/
│   └── xgb_model.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── explain_shap.py
│   ├── counterfactuals.py
│   ├── rules.py
│   ├── risk_timeline.py
│   └── app.py
│
├── outputs/
│   └── plots/
│
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone <repository-url>
cd sepsis-xai
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

### Windows
```bash
venv\Scripts\activate
```

### Linux/Mac
```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib streamlit joblib reportlab
```

---

# ▶️ Running the Project

## Run Streamlit Dashboard

```bash
streamlit run src/app.py
```

The application will open in your browser:

```bash
http://localhost:8501
```

---

# 📊 Dashboard Modules

## 🔴 Risk Prediction
Displays:
- Sepsis probability
- Risk category
- Confidence score

---

## 📈 SHAP Explanation
Visual explanation of:
- features increasing risk
- features decreasing risk

---

## 🔁 Treatment Simulation
Shows:
- suggested physiological improvements
- risk reduction after intervention

---

## ⏳ Future Risk Forecasting
Predicts:
- future risk progression
- critical threshold crossing

---

## 📜 Clinical Interpretation
Provides:
- medical reasoning
- human-readable explanations

---

# 🎯 Applications

- ICU monitoring systems
- Clinical decision support
- Early warning systems
- Explainable healthcare AI research

---

# ⚠️ Limitations

- Simulated treatment effects are heuristic-based
- Dataset imbalance may affect risk distribution
- Not intended for direct clinical deployment without validation

---

# 🔮 Future Work

- Real-time ICU integration
- Deep learning sequence models
- Personalized intervention planning
- Multi-hospital validation
- Reinforcement learning for treatment optimization

---

# 👥 Team Members

- Abdul Khader
- Neha Reddy
- Ansh V 

---

# 📚 References

1. PhysioNet Sepsis Challenge Dataset  
2. Lundberg et al. — SHAP: A Unified Approach to Interpreting Model Predictions  
3. Johnson et al. — MIMIC-III Clinical Database  
4. XGBoost Documentation  
5. Streamlit Documentation  

---

# 🏥 Conclusion

This project demonstrates how Explainable AI can improve trust and usability of machine learning systems in healthcare by combining prediction, explanation, simulation, and forecasting into a single clinical decision-support platform.