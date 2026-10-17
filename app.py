"""Public research demo using the adapted seven-feature logistic pipeline."""
from pathlib import Path
import csv

import streamlit as st

from inference import load_bundle, predict

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="CBC Research Demo", page_icon="🧬", layout="wide")


@st.cache_resource
def resources():
    return load_bundle()


model, config = resources()
st.title("CBC Research Demo")
st.caption("Routine blood-count modelling · Indian cohort research prototype")
st.warning("Research prototype. Not a diagnostic or screening test. Scores are not calibrated cancer probabilities.")
st.write("This demo uses the adapted logistic regression model with five measured CBC inputs and two calculated ratios.")
metric = config["held_out_metrics"]
columns = st.columns(4)
for column, label, value in zip(columns,
    ["Held-out AUC", "Held-out accuracy", "Sensitivity", "Specificity"],
    [f"{metric['auc']:.3f}", f"{metric['accuracy']:.1%}", f"{metric['sensitivity']:.1%}", f"{metric['specificity']:.1%}"]):
    column.metric(label, value)
st.caption("Internal nested cross-validation on 752 records with reconstructed labels. These metrics describe the development procedure, not an independent test of this deployed instance.")

input_tab, evidence_tab = st.tabs(["Enter CBC values", "Model evidence"])
with input_tab:
    st.info("Public research demo: use published sample measurements only. Do not enter private patient information.")
    st.write("Use the units shown. Enter no names, identifiers, or other personal details. Measurements are processed in memory and are not saved by this app.")
    with st.form("cbc_inputs"):
        left, right = st.columns(2)
        with left:
            wbc = st.number_input("WBC (10⁹/L)", min_value=0.0, value=None, step=0.01, key="wbc")
            rbc = st.number_input("RBC (10¹²/L)", min_value=0.0, value=None, step=0.01, key="rbc")
        with right:
            neut = st.number_input("Neutrophils (%)", min_value=0.0, max_value=100.0, value=None, step=0.1, key="neutrophils")
            lymph = st.number_input("Lymphocytes (%)", min_value=0.0, max_value=100.0, value=None, step=0.1, key="lymphocytes")
            mono = st.number_input("Monocytes (%)", min_value=0.0, max_value=100.0, value=None, step=0.1, key="monocytes")
        submitted = st.form_submit_button("Calculate research score", width="stretch")
    if submitted:
        try:
            result = predict(model, config, {"wbc": wbc, "rbc": rbc, "neutrophils": neut,
                                             "lymphocytes": lymph, "monocytes": mono})
            st.metric("Research model score", f"{result['score']:.3f}")
            position = "Above" if result["above_cutoff"] else "Below"
            st.info(f"{position} the research cutoff of {result['cutoff']:.3f}. This does not determine whether a person has cancer.")
            if result["undefined_ratios_imputed"]:
                st.caption("A ratio was undefined after rounding; the saved training median was used by the model pipeline.")
        except ValueError as error:
            st.error(str(error))
    with st.expander("How the inputs are processed"):
        st.write("Differential percentages are rounded consistently to whole percentages, matching the precision-harmonized experiment. NLR = neutrophils / lymphocytes; LMR = lymphocytes / monocytes. The saved pipeline then applies its training-fitted imputation, scaling, and logistic regression. Nothing is fitted from your inputs.")

with evidence_tab:
    st.subheader("Algorithm comparison")
    rows = list(csv.DictReader((ROOT / "bundle/results.csv").open()))
    table = [{"Algorithm": row["algorithm"], "AUC": round(float(row["auc"]), 3),
              "Accuracy (%)": round(float(row["accuracy"]) * 100, 1)}
             for row in rows if row["regime"] == "India only"]
    st.dataframe(table, hide_index=True, width="stretch")
    st.image(str(ROOT / "bundle/auc_comparison.png"), caption="Saved held-out results from the precision-harmonized comparison")
    st.write("Evaluation: 5 outer folds × 3 repeats, with 3 inner folds for tuning and threshold selection. Confidence intervals use participant-level bootstrap resampling. The final research pipeline is refitted on the development cohort; the displayed cutoff comes from training-side out-of-fold scores.")
    st.write("Limitations: labels are reconstructed from published supplementary records; cases and controls may have different hospital sources; treatment timing and source units need confirmation. Haemoglobin is excluded. An independent Indian cohort is required before clinical claims.")
    st.link_button("Original Indian hospital study", "https://www.nature.com/articles/s41598-023-27548-w")
st.caption(f"Model version: {config['version']} · CPU inference · No training runs in this app")
