"""Stateless JSON inference using the exact existing fitted model and cutoff."""
from contextlib import asynccontextmanager
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from inference import load_bundle, predict


@asynccontextmanager
async def lifespan(app):
    app.state.model, app.state.config = load_bundle()
    yield


app = FastAPI(
    title='CBC Model Inference API', version='1.0.0', lifespan=lifespan,
    description='Frozen Indian CBC logistic regression research pipeline. No training. '
                'Outputs are research scores, not diagnoses or calibrated cancer probabilities.',
)
origins = [value.strip() for value in os.getenv('ALLOWED_ORIGINS',
           'http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173').split(',') if value.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])


class CBCInput(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False,
                             json_schema_extra={'examples': [dict(wbc=8.95, rbc=4.41,
                                  neutrophils=65, lymphocytes=25, monocytes=9)]})
    wbc: float = Field(gt=0, description='White blood cells in 10^9/L')
    rbc: float = Field(gt=0, description='Red blood cells in 10^12/L')
    neutrophils: float = Field(ge=0, le=100, description='Neutrophils as a percentage')
    lymphocytes: float = Field(ge=0, le=100, description='Lymphocytes as a percentage')
    monocytes: float = Field(ge=0, le=100, description='Monocytes as a percentage')


class Prediction(BaseModel):
    model_version: str
    score: float
    cutoff: float
    above_cutoff: bool
    undefined_ratios_imputed: bool
    research_only: bool = True


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, error: RequestValidationError):
    # Return useful schema errors without echoing submitted measurements or extra fields.
    return JSONResponse(status_code=422, content={'detail': [
        {'field': '.'.join(str(part) for part in item['loc']), 'message': item['msg']}
        for item in error.errors()]})


@app.get('/')
def index():
    return {'service': 'CBC model inference', 'docs': '/docs', 'health': '/health',
            'prediction': 'POST /predict', 'research_only': True}


@app.get('/health')
def health(request: Request):
    return {'status': 'ok', 'model_version': request.app.state.config['version']}


@app.post('/predict', response_model=Prediction)
def prediction(values: CBCInput, request: Request):
    try:
        result = predict(request.app.state.model, request.app.state.config, values.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    return Prediction(model_version=request.app.state.config['version'], **result)
