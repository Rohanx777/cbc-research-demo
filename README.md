# CBC Research Demo

Streamlit interface for a frozen Indian-cohort CBC logistic regression research pipeline.
Research use only. Not a diagnostic or screening test. Scores are not calibrated cancer probabilities.

## Run

Use Python 3.12 and the pinned requirements:

```sh
pip install -r requirements.txt
streamlit run app.py
```

On Streamlit Community Cloud select branch `main`, entrypoint `app.py`, and Python 3.12.
No API keys, cloud credentials or training datasets are required.

## Model and evaluation

Five measured CBC values become seven features using the original experiment's whole-percentage
rounding and NLR/LMR derivation. The saved pipeline supplies training-fitted imputation, scaling
and logistic regression. The app never fits a model or persists submitted measurements.

The final model was refitted on the Indian development cohort. Displayed metrics come from
internal nested cross-validation on 752 records with reconstructed case/control labels:
AUC 0.713, accuracy 63.2%, sensitivity 35.9%, specificity 90.4%. These do not independently
validate this final refitted model. Hospital-source confounding, differential precision and
label reconstruction limit interpretation. Haemoglobin is excluded due to uncertain units.

Source study: https://www.nature.com/articles/s41598-023-27548-w

## Verify

Use published sample measurements WBC 8.95 (10^9/L), RBC 4.41 (10^12/L), neutrophils 65%,
lymphocytes 25%, monocytes 9%. The expected score is 0.653 and research cutoff 0.627.
This check is for software consistency, not diagnosis. Use published sample measurements only.

Only inference artifacts and aggregate results are included. No patient-level datasets,
identifiers, credentials or training scripts are included.
