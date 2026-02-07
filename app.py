import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load   # ← ESTA LÍNEA FALTABA
# App config (SOLO UNA VEZ y lo primero)
st.set_page_config(page_title="IVF/ICSI Predictor (Table S2)", layout="wide")

st.title("IVF/ICSI Cumulative Live Birth Predictor (Table S2) — Demo App")
st.caption("⚠️ Demo only. Synthetic model/data; not clinically valid.")
st.sidebar.write("Python:", sys.version)

DEFAULT_MODEL_PATH = "xgb_table_s2.joblib"

LABELS = {
    "age_years": "Age (years)",
    "bmi_kg_m2": "BMI (kg/m²)",
    "infertility_duration_years": "Infertility duration (years)",
    "num_previous_ivf": "Number of previous IVF cycles",
    "lh0_iu_l": "LH0 basal (IU/L)",
    "e20_pg_ml": "E20 basal estradiol (pg/mL)",
    "prl0_ng_ml": "PRL0 basal prolactin (ng/mL)",
    "fsh0_iu_l": "FSH0 basal (IU/L)",
    "t0_ng_ml": "T0 basal testosterone (ng/mL)",
    "e21_pg_ml": "E21 post-trigger estradiol (pg/mL)",
    "prl1_ng_ml": "PRL1 post-trigger prolactin (ng/mL)",
    "lh1_iu_l": "LH1 post-trigger (IU/L)",
    "fsh1_iu_l": "FSH1 post-trigger (IU/L)",
    "t1_ng_ml": "T1 post-trigger testosterone (ng/mL)",
    "p1_ng_ml": "P1 post-trigger progesterone (ng/mL)",
    "total_fsh_dose": "Total FSH dose",
    "total_hmg_dose": "Total HMG dose",
}


# -----------------------------
# Helpers
# -----------------------------
def cumulative_success_curve(p_cycle: float, max_cycles: int) -> pd.DataFrame:
    cycles = np.arange(1, max_cycles + 1)
    cum = 1 - (1 - p_cycle) ** cycles
    return pd.DataFrame(
        {"cycle_number": cycles, "per_cycle_probability": p_cycle, "cumulative_probability": cum}
    )


@st.cache_resource
def load_model_bundle(model_path: str):
    """
    Expects a joblib bundle dict with:
      - pipeline (sklearn Pipeline: imputer -> xgb model)
      - features (list)
      - target (str)
    """
    bundle = load(model_path)
    pipe = bundle["pipeline"]
    features = bundle.get("features", None)
    target = bundle.get("target", "cumulative_live_birth")
    return bundle, pipe, features, target


def build_input_form(features):
    """
    Returns a single-row DataFrame with columns=features.
    Missing values are np.nan (imputer handles them).
    """
    st.subheader("1) Enter patient variables (leave blank if unknown)")

    cols = st.columns(3)
    values = {}

    # Streamlit numeric_input can't be blank; so we use text_input and parse.
    def numeric_text_input(key, label, column):
        raw = column.text_input(label, value="", key=key, placeholder="blank = missing")
        raw = raw.strip()
        if raw == "":
            return np.nan
        try:
            return float(raw)
        except ValueError:
            st.warning(f"Invalid number for '{label}'. Treating as missing.")
            return np.nan

    for i, f in enumerate(features):
        col = cols[i % 3]
        values[f] = numeric_text_input(f"inp_{f}", LABELS.get(f, f), col)

    x = pd.DataFrame([values], columns=features)
    return x


def shap_explain_one(pipe, x_raw: pd.DataFrame, features):
    """
    Returns:
      - shap values row (1, n_features) in model output space (log-odds for XGB binary)
      - base value
      - imputed row values
    """
    imputer = pipe.named_steps["imputer"]
    model = pipe.named_steps["model"]

    x_imp = imputer.transform(x_raw)  # numpy array (1, n_features)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_imp)

    # Handle SHAP output versions
    if isinstance(shap_values, list):
        # sometimes returns [class0, class1] for binary
        shap_row = np.array(shap_values[-1])[0]
    else:
        shap_row = np.array(shap_values)[0]

    base = explainer.expected_value
    if isinstance(base, (list, tuple, np.ndarray)):
        base = float(np.array(base).ravel()[-1])
    else:
        base = float(base)

    return shap_row, base, x_imp[0]


def shap_rank_table(shap_row, x_imp_row, features):
    df = pd.DataFrame(
        {
            "Feature": [LABELS.get(f, f) for f in features],
            "Value used (imputed if missing)": x_imp_row,
            "SHAP contribution": shap_row,
            "Abs(SHAP)": np.abs(shap_row),
            "Direction": np.where(shap_row >= 0, "↑ increases", "↓ decreases"),
        }
    ).sort_values("Abs(SHAP)", ascending=False)
    return df.drop(columns=["Abs(SHAP)"])


# -----------------------------
# Sidebar: model loading
# -----------------------------
st.sidebar.header("Settings")

model_path = st.sidebar.text_input("Model path", value=DEFAULT_MODEL_PATH)
cycles = st.sidebar.slider("Cycles for cumulative curve (K)", min_value=1, max_value=8, value=4)
top_n = st.sidebar.slider("Top SHAP features to show", min_value=5, max_value=17, value=10)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: Put `xgb_table_s2.joblib` in the same folder as this app, "
    "or set the path above."
)

# Try to load model
try:
    bundle, pipe, FEATURES, target = load_model_bundle(model_path)
    # 🔧 Fix compatibilidad XGBoost antiguos
model = pipe.named_steps["model"]
if not hasattr(model, "use_label_encoder"):
    model.use_label_encoder = False

except Exception as e:
    st.error(
        "Could not load model. Make sure the file exists and is a joblib bundle.\n\n"
        f"Details: {e}"
    )
    st.stop()

if not FEATURES:
    st.error("Model bundle does not include `features`. Re-save the bundle with features list.")
    st.stop()

# Show small metadata
with st.expander("Model info"):
    st.write(f"Target: `{target}`")
    st.write(f"Number of features: {len(FEATURES)}")
    note = bundle.get("note", "")
    if note:
        st.warning(note)

# -----------------------------
# Main: input + predict
# -----------------------------
x = build_input_form(FEATURES)

predict_clicked = st.button("2) Predict patient success", type="primary")

if predict_clicked:
    # Predict
    try:
        proba = float(pipe.predict_proba(x)[0, 1])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("2) Patient expected success")
    st.metric("Per-cycle probability of cumulative live birth", f"{proba:.1%}", help="Predicted probability for one stimulation cycle.")

    # Cumulative curve
    st.subheader("3) Expected cumulative success curve")
    st.caption("Assumes independent cycles with the same per-cycle probability p:  P(success by K) = 1 - (1 - p)^K")

    curve = cumulative_success_curve(proba, cycles)
    st.dataframe(curve.style.format({"per_cycle_probability": "{:.3f}", "cumulative_probability": "{:.3f}"}), use_container_width=True)

    # Plot curve
    fig = plt.figure()
    plt.plot(curve["cycle_number"], curve["cumulative_probability"], marker="o")
    plt.ylim(0, 1)
    plt.xlabel("Cycle number (K)")
    plt.ylabel("P(≥1 live birth by K)")
    plt.title("Expected cumulative success curve")
    st.pyplot(fig)
    plt.close(fig)

    # SHAP
    st.subheader("4) SHAP explanation (per patient)")
    st.caption("SHAP values show how each feature pushes the model prediction up or down relative to the baseline.")

    try:
        shap_row, base_value, x_imp_row = shap_explain_one(pipe, x, FEATURES)
    except Exception as e:
        st.error(
            "SHAP explanation failed. Ensure `shap` is installed and the model is an XGBoost tree model.\n\n"
            f"Details: {e}"
        )
        st.stop()

    # Ranked contributions table
    shap_df = shap_rank_table(shap_row, x_imp_row, FEATURES)
    st.dataframe(shap_df.head(top_n), use_container_width=True)

    # Waterfall plot
    st.markdown("**Waterfall plot (top contributions)**")
    try:
        # Create SHAP Explanation object for plotting
        exp = shap.Explanation(
            values=shap_row,
            base_values=base_value,
            data=x_imp_row,
            feature_names=[LABELS.get(f, f) for f in FEATURES],
        )
        fig2 = plt.figure()
        shap.plots.waterfall(exp, max_display=top_n, show=False)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
    except Exception as e:
        st.warning(f"Could not render SHAP waterfall plot: {e}")

    st.success("Done.")
