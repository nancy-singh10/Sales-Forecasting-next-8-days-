"""
Sales Forecasting API — Production Backend Service
====================================================
A production-grade REST API for beverage sales forecasting across US states.

Architecture:
    - Models are preloaded into memory at startup (not per-request)
    - Predictions are cached with TTL to avoid redundant computation
    - Structured JSON logging for observability (ELK/CloudWatch compatible)
    - API versioned under /api/v1
    - Rate limiting per client IP
    - Health checks include dependency readiness (model availability)
    - Graceful startup/shutdown lifecycle

Endpoints:
    GET  /                          → Root redirect info
    GET  /health                    → Liveness probe
    GET  /ready                     → Readiness probe (checks model availability)
    GET  /api/v1/forecast/{state}   → 8-week sales forecast for a state
    GET  /api/v1/states             → List available states
    GET  /api/v1/model-info         → Current model metadata
"""

import sys
import os
import time
import json
import logging
import uuid
import hashlib
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from functools import lru_cache
from collections import defaultdict

# ─── PATH SETUP ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib

from predict import predict_next_8_weeks


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class Settings:
    """Application settings — reads from env vars with sensible defaults."""
    APP_NAME: str = os.getenv("APP_NAME", "Sales Forecasting API")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
    RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

    # Cache
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5 min
    CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "200"))

    # Model paths
    MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")
    BEST_MODEL_PATH: str = os.path.join(MODELS_DIR, "best_model_name.pkl")


settings = Settings()


# ═══════════════════════════════════════════════════════════════
#  STRUCTURED JSON LOGGING
# ═══════════════════════════════════════════════════════════════

class JSONFormatter(logging.Formatter):
    """Produces structured JSON log lines (ELK / CloudWatch compatible)."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Merge any extra fields attached to the record
        for key in ("request_id", "client_ip", "method", "path",
                     "status_code", "duration_ms", "state"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry)


def _setup_logging():
    handler_console = logging.StreamHandler()
    handler_console.setFormatter(JSONFormatter())

    handler_file = logging.FileHandler(
        os.path.join(PROJECT_ROOT, "api_requests.log"), mode="a"
    )
    handler_file.setFormatter(JSONFormatter())

    root = logging.getLogger("ForecastAPI")
    root.setLevel(getattr(logging, settings.LOG_LEVEL))
    root.handlers.clear()
    root.addHandler(handler_console)
    root.addHandler(handler_file)
    return root


logger = _setup_logging()


# ═══════════════════════════════════════════════════════════════
#  IN-MEMORY MODEL REGISTRY  (loaded once at startup)
# ═══════════════════════════════════════════════════════════════

class ModelRegistry:
    """
    Holds pre-loaded model artifacts in memory.
    Avoids hitting disk on every prediction request.
    """

    def __init__(self):
        self.best_model_name: Optional[str] = None
        self._loaded: bool = False
        self._load_error: Optional[str] = None

    def load(self):
        """Load model metadata at startup. Heavy models stay on disk
        and are loaded by predict.py — we only cache the *name* here."""
        try:
            self.best_model_name = joblib.load(settings.BEST_MODEL_PATH)
            self._loaded = True
            logger.info("Model registry loaded — best model: %s", self.best_model_name)
        except FileNotFoundError:
            self.best_model_name = "xgboost"
            self._loaded = True
            self._load_error = "best_model_name.pkl not found; defaulting to xgboost"
            logger.warning(self._load_error)
        except Exception as e:
            self._load_error = str(e)
            logger.error("Failed to load model registry: %s", e, exc_info=True)

    @property
    def is_ready(self) -> bool:
        return self._loaded

    @property
    def status_detail(self) -> str:
        if self._loaded:
            return "models loaded"
        return self._load_error or "not initialized"


model_registry = ModelRegistry()


# ═══════════════════════════════════════════════════════════════
#  PREDICTION CACHE  (TTL-based in-memory cache)
# ═══════════════════════════════════════════════════════════════

class PredictionCache:
    """Simple TTL cache for forecast results.
    Avoids re-running heavy model inference for repeated requests."""

    def __init__(self, ttl: int = 300, max_size: int = 200):
        self._cache: Dict[str, dict] = {}
        self._ttl = ttl
        self._max_size = max_size
        self.hits = 0
        self.misses = 0

    def _key(self, state: str, model: str) -> str:
        return hashlib.md5(f"{state}:{model}".encode()).hexdigest()

    def get(self, state: str, model: str) -> Optional[List[float]]:
        key = self._key(state, model)
        entry = self._cache.get(key)
        if entry and (time.time() - entry["ts"]) < self._ttl:
            self.hits += 1
            return entry["data"]
        if entry:
            del self._cache[key]  # expired
        self.misses += 1
        return None

    def put(self, state: str, model: str, data: List[float]):
        if len(self._cache) >= self._max_size:
            # evict oldest entry
            oldest = min(self._cache, key=lambda k: self._cache[k]["ts"])
            del self._cache[oldest]
        key = self._key(state, model)
        self._cache[key] = {"data": data, "ts": time.time()}

    @property
    def stats(self) -> dict:
        return {
            "size": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hits / max(self.hits + self.misses, 1) * 100:.1f}%",
        }


cache = PredictionCache(
    ttl=settings.CACHE_TTL_SECONDS,
    max_size=settings.CACHE_MAX_SIZE,
)


# ═══════════════════════════════════════════════════════════════
#  RATE LIMITER  (sliding window per client IP)
# ═══════════════════════════════════════════════════════════════

class RateLimiter:
    """In-memory sliding-window rate limiter keyed by client IP."""

    def __init__(self, max_requests: int, window_seconds: int):
        self._max = max_requests
        self._window = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - self._window
        # Prune old timestamps
        self._requests[client_ip] = [
            ts for ts in self._requests[client_ip] if ts > window_start
        ]
        if len(self._requests[client_ip]) >= self._max:
            return False
        self._requests[client_ip].append(now)
        return True

    @property
    def remaining(self):
        """Helper used after is_allowed to return remaining count."""
        return self._max


rate_limiter = RateLimiter(
    max_requests=settings.RATE_LIMIT_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
)


# ═══════════════════════════════════════════════════════════════
#  PYDANTIC RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    service: str
    version: str
    environment: str
    uptime_seconds: float
    timestamp: str


class ReadinessResponse(BaseModel):
    status: str = Field(..., example="ready")
    models_loaded: bool
    models_detail: str
    cache_stats: dict
    timestamp: str


class ForecastResponse(BaseModel):
    request_id: str = Field(..., description="Unique request identifier for tracing")
    state: str
    forecast_weeks: int = 8
    forecast: List[float] = Field(..., description="Predicted sales values for the next 8 weeks")
    model_used: str
    cached: bool = Field(False, description="Whether the result was served from cache")
    generated_at: str


class StatesResponse(BaseModel):
    count: int
    states: List[str]


class ModelInfoResponse(BaseModel):
    best_model: str
    all_models: List[str]
    selection_criteria: str
    training_data: str
    forecast_horizon: int = 8


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None
    timestamp: str


# ═══════════════════════════════════════════════════════════════
#  VALID STATES
# ═══════════════════════════════════════════════════════════════

VALID_STATES = [
    "Alabama", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Florida", "Georgia", "Illinois", "Indiana",
    "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
    "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
    "Washington", "West Virginia", "Wisconsin", "Wyoming"
]

_VALID_STATES_SET = set(VALID_STATES)   # O(1) lookup


# ═══════════════════════════════════════════════════════════════
#  APP FACTORY
# ═══════════════════════════════════════════════════════════════

_start_time = time.time()

app = FastAPI(
    title=settings.APP_NAME,
    description=__doc__,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "Nancy Singh"},
    license_info={"name": "MIT"},
    responses={
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        404: {"model": ErrorResponse, "description": "State not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── LIFECYCLE EVENTS ───────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    """Pre-load models and warm up resources on application boot."""
    logger.info("=== SERVICE STARTING === env=%s version=%s",
                settings.ENVIRONMENT, settings.APP_VERSION)
    model_registry.load()
    logger.info("=== SERVICE READY ===")


@app.on_event("shutdown")
async def on_shutdown():
    """Cleanup on graceful shutdown."""
    logger.info("=== SERVICE SHUTTING DOWN === cache_stats=%s", cache.stats)


# ─── REQUEST / RESPONSE MIDDLEWARE ───────────────────────────

@app.middleware("http")
async def request_pipeline(request: Request, call_next):
    """
    Central middleware that handles:
      1. Request ID generation & propagation
      2. Rate limiting
      3. Request/response structured logging
      4. Timing
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    client_ip = request.client.host if request.client else "unknown"
    request.state.request_id = request_id

    # ── Rate limiting ──
    if not rate_limiter.is_allowed(client_ip):
        logger.warning("Rate limit exceeded",
                        extra={"request_id": request_id, "client_ip": client_ip})
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "detail": f"Max {settings.RATE_LIMIT_REQUESTS} requests "
                          f"per {settings.RATE_LIMIT_WINDOW_SECONDS}s. Try again later.",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            headers={
                "X-Request-ID": request_id,
                "Retry-After": str(settings.RATE_LIMIT_WINDOW_SECONDS),
            },
        )

    # ── Execute request ──
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 2)

    # ── Log ──
    logger.info(
        "%s %s → %d (%.1fms)",
        request.method, request.url.path,
        response.status_code, duration_ms,
        extra={
            "request_id": request_id,
            "client_ip": client_ip,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    # ── Attach tracing headers ──
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(duration_ms)
    return response


# ═══════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════

# ─── Infrastructure ──────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Root redirect — points callers to the docs."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "api": "/api/v1/forecast/{state}",
    }


@app.get("/health", response_model=HealthResponse, tags=["Infrastructure"])
def health_check():
    """
    **Liveness probe** — returns 200 if the process is alive.
    Used by container orchestrators (Docker / K8s) to detect crashes.
    """
    return HealthResponse(
        status="healthy",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        uptime_seconds=round(time.time() - _start_time, 1),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/ready", response_model=ReadinessResponse, tags=["Infrastructure"])
def readiness_check():
    """
    **Readiness probe** — returns 200 only when models are loaded and
    the service can actually serve predictions. K8s will not route traffic
    until this returns 200.
    """
    is_ready = model_registry.is_ready
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content=ReadinessResponse(
            status="ready" if is_ready else "not_ready",
            models_loaded=is_ready,
            models_detail=model_registry.status_detail,
            cache_stats=cache.stats,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ).model_dump(),
    )


# ─── API v1 — Forecasting ────────────────────────────────────

@app.get(
    "/api/v1/forecast/{state}",
    response_model=ForecastResponse,
    tags=["Forecasting"],
    summary="Generate 8-week sales forecast for a US state",
)
def forecast(state: str, request: Request):
    """
    Generate an **8-week beverage sales forecast** for the specified US state.

    - **state**: Exact state name (e.g. `Texas`, `California`).
      Use `GET /api/v1/states` to see all valid names.
    - Results are **cached for 5 minutes** to reduce inference latency on repeated calls.
    - The model used is the one with the **lowest RMSE** on the validation set.
    """
    request_id = getattr(request.state, "request_id", "N/A")
    model_name = model_registry.best_model_name or "xgboost"

    # ── Validate ──
    if state not in _VALID_STATES_SET:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "state_not_found",
                "detail": f"'{state}' is not a valid state. "
                          f"Use GET /api/v1/states for the full list.",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # ── Check cache ──
    cached_result = cache.get(state, model_name)
    if cached_result is not None:
        logger.info("Cache HIT for %s", state,
                     extra={"request_id": request_id, "state": state})
        return ForecastResponse(
            request_id=request_id,
            state=state,
            forecast=cached_result,
            model_used=model_name,
            cached=True,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    # ── Run inference ──
    logger.info("Cache MISS — running inference for %s", state,
                 extra={"request_id": request_id, "state": state})
    try:
        preds = predict_next_8_weeks(state)
    except Exception as e:
        logger.error("Prediction failed for %s: %s", state, e,
                      extra={"request_id": request_id}, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "prediction_error",
                "detail": f"Model inference failed for '{state}': {str(e)}",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # Handle error dict returned by predict module
    if isinstance(preds, dict) and "error" in preds:
        raise HTTPException(status_code=500, detail=preds)

    # ── Cache result ──
    cache.put(state, model_name, preds)

    return ForecastResponse(
        request_id=request_id,
        state=state,
        forecast=preds,
        model_used=model_name,
        cached=False,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ─── Backward-compatible /forecast/{state} redirect ─────────

@app.get("/forecast/{state}", include_in_schema=False)
def forecast_legacy(state: str, request: Request):
    """Legacy endpoint — proxies to /api/v1/forecast/{state}."""
    return forecast(state, request)


# ─── API v1 — Metadata ───────────────────────────────────────

@app.get("/api/v1/states", response_model=StatesResponse, tags=["Metadata"])
def list_states():
    """Returns all US states available for forecasting."""
    return StatesResponse(count=len(VALID_STATES), states=VALID_STATES)


@app.get("/api/v1/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
def model_info():
    """Returns metadata about the currently active forecasting model."""
    return ModelInfoResponse(
        best_model=model_registry.best_model_name or "xgboost",
        all_models=["sarima", "prophet", "xgboost", "lstm"],
        selection_criteria="Lowest RMSE on 8-week validation set",
        training_data="US Beverage Sales (2019–2023, weekly)",
        forecast_horizon=8,
    )


# ═══════════════════════════════════════════════════════════════
#  GLOBAL EXCEPTION HANDLER
# ═══════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all — ensures unhandled errors always return structured JSON."""
    request_id = getattr(request.state, "request_id", "N/A")
    logger.critical("Unhandled exception: %s", exc,
                     extra={"request_id": request_id}, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": "An unexpected error occurred. Check logs for details.",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        headers={"X-Request-ID": request_id},
    )
