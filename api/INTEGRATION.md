# Model API integration

The model runs behind `POST /predict`. The Streamlit page is a separate interface;
do not fetch its HTML URL to request predictions. Use the API service's deployed
base URL, which must be recorded after a successful cloud deployment.

## Input and output

```json
{"wbc":8.95,"rbc":4.41,"neutrophils":65,"lymphocytes":25,"monocytes":9}
```

WBC units are 10^9/L; RBC units are 10^12/L. Differentials are percentages, not
absolute counts, and their sum must not exceed 100. Send all five as JSON numbers.
Do not send names, identifiers, age, haemoglobin or the dataset. The server rounds
differentials consistently, derives NLR/LMR and uses the saved preprocessing.

For the published example the expected JSON is:

```json
{
  "model_version":"india-cbc-logistic-c87ec8485159",
  "score":0.6528833817519397,
  "cutoff":0.626816175626443,
  "above_cutoff":true,
  "undefined_ratios_imputed":false,
  "research_only":true
}
```

The score is not a calibrated cancer probability and the threshold is not a diagnosis.
The API performs no fitting, data writes, or prediction-body logging. Hosting-provider
infrastructure may retain request metadata; use published sample measurements for the hackathon.

## JavaScript or TypeScript

```javascript
export async function getModelResult(apiBaseUrl, measurements) {
  const response = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(measurements),
    signal: AbortSignal.timeout(120000),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(JSON.stringify(body.detail));
  return body;
}
```

Configure the actual frontend origins in the Render `ALLOWED_ORIGINS` environment
variable, separated by commas, for direct browser calls. Backend-to-backend calls
do not require CORS. Never put a future private API key in frontend source.
For an application backend, store the endpoint in a server environment variable.

## Deploy on Render Free

Repository: https://github.com/Rohanx777/cbc-research-demo

Create a Python web service from `main` using `pip install -r api/requirements.txt`
as the build command and `uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1 --no-access-log`
as the start command. Keep the root directory blank. Use Python 3.12.12, the Free
instance and health check `/health`. `render.yaml` provides the same settings.

Once running, verify `GET /health`, open `/docs`, and send the published input to
`POST /predict`. Compare the returned numeric score to the expected value above.
Free services sleep after 15 minutes of inactivity and can take about a minute to
wake. Show a loading state, handle failures and warm up with `/health` before judging.

## Local API backup

From the repository root after installing `api/requirements.txt`:

```sh
uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

Use `http://127.0.0.1:8000/docs` for the interactive request schema and test client.
This API is a research prototype with no built-in authentication or rate limiter.
It is intended for the hackathon, not clinical production deployment.
