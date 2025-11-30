#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PRC 2025 – Fuel burn (kg/min) XGBoost + BayesOpt + Submission + S3 (MinIO)

Changelog vs previous attempts:
- Robust dtype handling for pandas >=2.x (no BooleanDtype errors)
- Memory-safe one-hot (sparse) with deterministic column order
- Best-iteration predictions via iteration_range (XGBoost 2+)
- BayesOpt API autodetect:
    * new: AcquisitionFunction + maximize(acquisition_function=...)
    * old: acq='ei', kappa/xi
- All paths are relative to current directory ("./")
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from sklearn.utils import check_random_state

# Optional deps (handled below)
try:
    from bayes_opt import BayesianOptimization  # type: ignore
    _HAS_BAYESOPT = True
except Exception:
    _HAS_BAYESOPT = False

try:
    # For writing parquet like the R script
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    _HAS_ARROW = True
except Exception:
    _HAS_ARROW = False

# ----------------------------
# Paths (all relative to ./)
# ----------------------------
BASE = Path(".").resolve()
IN_FEATURES = BASE / "features_intervals.csv"
IN_SUBMISSION = BASE / "submission_intervals.csv"
IN_TAXI_SCORED = BASE / "submission_intervals_v4_scored.csv"  # taxi preds

OUT_MODEL_JSON = BASE / "xgb_fuel_burn_final_model.json"
OUT_META_JSON = BASE / "xgb_fuel_burn_metadata.json"
OUT_IMPORTANCE_CSV = BASE / "importance_matrix.csv"
OUT_PRED_ERR_CSV = BASE / "predicted_vs_actual_full.csv"
OUT_PARQUET = BASE / "honest-rose_v44.parquet"

# ----------------------------
# Small utilities
# ----------------------------
def log(msg: str) -> None:
    print(msg, flush=True)

def robust_is_numeric(s: pd.Series) -> bool:
    import pandas.api.types as ptypes
    return ptypes.is_integer_dtype(s) or ptypes.is_float_dtype(s)

def normalize_columns_for_model(df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    """
    - Boolean & pandas BooleanDtype -> bool with NaN->False
    - Strings/categoricals -> string dtype with NA sentinel
    - Numerics -> leave as-is (NaN allowed; DMatrix handles missing)
    """
    out = df.copy()
    for c in feat_cols:
        s = out[c]
        if s.dtype.name == "boolean":  # pandas BooleanDtype
            out[c] = s.fillna(False).astype(bool)
        elif s.dtype == bool:
            out[c] = s.fillna(False).astype(bool)
        elif robust_is_numeric(s):
            # keep NaN as-is; xgboost DMatrix handles missing
            out[c] = pd.to_numeric(s, errors="coerce")
        else:
            # normalize to string with NA sentinel
            out[c] = s.astype("string").fillna("__NA__")
    return out

def one_hot_sparse(df: pd.DataFrame,
                   numeric_cols: List[str]) -> Tuple[sp.csr_matrix, List[str]]:
    """
    Return CSR matrix with numeric (first) + one-hot (sparse) for all non-numeric cols.
    """
    all_cols = list(df.columns)
    obj_cols = [c for c in all_cols if c not in numeric_cols]

    # Ensure booleans aren't mis-detected; cast to string with sentinel? no: keep as 0/1
    # Convert booleans to 0/1 numeric to avoid sparse-dtype pitfalls
    bool_like = [c for c in obj_cols if df[c].dtype == bool]
    if bool_like:
        for c in bool_like:
            df[c] = df[c].astype(np.uint8)
        # move them into numeric group
        numeric_cols = numeric_cols + bool_like
        obj_cols = [c for c in obj_cols if c not in bool_like]

    # Numeric block (float32)
    X_num = sp.csr_matrix(df[numeric_cols].astype(np.float32).to_numpy()) if numeric_cols else sp.csr_matrix((len(df), 0), dtype=np.float32)

    # Categorical block: get_dummies(sparse=True) with explicit dtype to avoid BooleanDtype
    if obj_cols:
        cats = pd.get_dummies(df[obj_cols], sparse=True, dtype=np.uint8, dummy_na=False)
        # Some pandas versions need .sparse.to_coo(); guard if empty
        if cats.shape[1] > 0:
            X_cat = cats.sparse.to_coo().tocsr()
            cat_names = list(cats.columns)
        else:
            X_cat = sp.csr_matrix((len(df), 0), dtype=np.uint8)
            cat_names = []
    else:
        X_cat = sp.csr_matrix((len(df), 0), dtype=np.uint8)
        cat_names = []

    X = sp.hstack([X_num, X_cat], format="csr")
    names = list(numeric_cols) + cat_names
    return X, names

def align_sparse_to_ref(X: sp.csr_matrix, names: List[str], ref_names: List[str], dtype=np.float32) -> sp.csr_matrix:
    """
    Reorder / pad columns so X has columns in ref_names order.
    Extra columns are dropped; missing columns are zero-filled.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    cols = []
    for n in ref_names:
        idx = name_to_idx.get(n, None)
        if idx is None:
            cols.append(sp.csr_matrix((X.shape[0], 1), dtype=dtype))
        else:
            col = X[:, idx]
            if col.dtype != dtype:
                col = col.astype(dtype)
            cols.append(col)
    return sp.hstack(cols, format="csr")

def make_group_folds(groups: np.ndarray, K: int, rng: np.random.RandomState):
    """
    Return a list of (train_idx, test_idx) tuples for xgboost.cv.
    'groups' must be an array with one entry per training row.
    """
    uniq = pd.unique(groups)
    rng.shuffle(uniq)
    splits = np.array_split(uniq, K)

    folds = []
    idx_all = np.arange(groups.shape[0])
    for k in range(K):
        test_mask = np.isin(groups, splits[k])
        test_idx = idx_all[test_mask]
        train_idx = idx_all[~test_mask]
        folds.append((train_idx, test_idx))
    return folds

def pred_with_best(model: xgb.Booster, dmat: xgb.DMatrix, best_iter: Optional[int]) -> np.ndarray:
    """
    Use best_iteration with iteration_range; end is exclusive, so pass (0, best_iter).
    """
    bi = int(best_iter) if best_iter is not None else int(getattr(model, "best_iteration", 0) or 0)
    if bi <= 0:
        return model.predict(dmat)
    # iteration_range end is exclusive
    return model.predict(dmat, iteration_range=(0, bi))

def gpu_params() -> Dict[str, object]:
    # XGBoost >= 2.0: GPU => device='cuda' + tree_method='hist'
    # Fall back to CPU if CUDA not available at runtime (handled where used).
    return dict(tree_method="hist", device="cuda")

def cpu_params() -> Dict[str, object]:
    return dict(tree_method="hist", device="cpu")
    
def get_best_iter_and_valrmse(model, dvalid, y_valid):
    """
    Robustly obtain (best_iteration, val_rmSE).
    Works even if booster has no evals_result() or history wasn't serialized.
    """
    # 1) If early stopping recorded best_iteration, try to read matching RMSE from history
    bi = getattr(model, "best_iteration", None)
    if isinstance(bi, (int, np.integer)) and bi and bi > 0:
        # try history first
        try:
            er = model.evals_result()
            if er and "eval" in er and "rmse" in er["eval"]:
                return int(bi), float(er["eval"]["rmse"][int(bi) - 1])
        except Exception:
            pass
        # fallback: compute RMSE with predictions up to best_iteration
        try:
            pred = model.predict(dvalid, iteration_range=(0, int(bi)))
        except TypeError:
            pred = model.predict(dvalid, ntree_limit=int(bi))
        rmse = float(np.sqrt(np.mean((pred - y_valid) ** 2)))
        return int(bi), rmse

    # 2) No best_iteration recorded → try to infer from history
    try:
        er = model.evals_result()
        if er and "eval" in er and "rmse" in er["eval"]:
            bi_hist = int(np.argmin(er["eval"]["rmse"])) + 1
            rmse = float(np.min(er["eval"]["rmse"]))
            return bi_hist, rmse
    except Exception:
        pass

    # 3) Last resort: use all trees; compute RMSE directly
    try:
        bi_all = int(model.num_boosted_rounds())
    except Exception:
        bi_all = 0
    pred = model.predict(dvalid)  # full model
    rmse = float(np.sqrt(np.mean((pred - y_valid) ** 2)))
    return bi_all, rmse

# ----------------------------
# Load & prepare data
# ----------------------------
if not IN_FEATURES.exists():
    raise FileNotFoundError(f"Missing {IN_FEATURES}")

log("Loading features_intervals.csv …")
all_df = (
    pd.read_csv(IN_FEATURES)
    .assign(
        status=lambda d: np.where(d["pct_elapsed_mid"] < 0, "taxi_out",
                 np.where(d["pct_elapsed_mid"] > 100, "taxi_in", "inflight"))
    )
)
all_df = all_df[~all_df["status"].isin(["taxi_in", "taxi_out"])].copy()

if "fuel_kg_min" not in all_df.columns:
    raise ValueError("Column 'fuel_kg_min' is required")

all_df = all_df[~all_df["fuel_kg_min"].isna()].copy()

# Drop constant or all-NA columns
drop_cols_const = [c for c in all_df.columns
                   if (all_df[c].isna().all()) or (all_df[c].nunique(dropna=False) <= 1)]
if drop_cols_const:
    all_df = all_df.drop(columns=drop_cols_const)

drop_id_cols = [
    "idx", "flight_id", "start", "end", "flight_date", "takeoff", "landed",
    "start_hour_utc", "end_hour_utc", "midpoint_utc", "model_time_utc",
    "points_file_exists", "origin_icao", "dest_icao", "dow", "month",
    "weather_code_text", "precipitation", "origin_region", "dest_region",
    "status"
]
keep_cols = [c for c in all_df.columns if c not in drop_id_cols]
if "fuel_kg_min" not in keep_cols:
    raise ValueError("Column 'fuel_kg_min' got dropped unexpectedly")

rng = check_random_state(1337)
all_flights = pd.unique(all_df["flight_id"])
test_flights = rng.choice(all_flights, size=int(0.20 * len(all_flights)), replace=False)

train_df = all_df[~all_df["flight_id"].isin(test_flights)][keep_cols].reset_index(drop=True)
test_df  = all_df[ all_df["flight_id"].isin(test_flights)][keep_cols].reset_index(drop=True)

# Keep flight_id arrays (for group folds, error table later)
tr_fids = all_df.loc[~all_df["flight_id"].isin(test_flights), "flight_id"].values
te_fids = all_df.loc[ all_df["flight_id"].isin(test_flights), "flight_id"].values

# ----------------------------
# Build sparse matrices
# ----------------------------
feat_cols = [c for c in train_df.columns if c != "fuel_kg_min"]
train_df_norm = normalize_columns_for_model(train_df, feat_cols)
test_df_norm  = normalize_columns_for_model(test_df, feat_cols)

# numeric = robust numeric + booleans we later cast; we detect using pandas api
import pandas.api.types as ptypes
numeric_cols = [c for c in feat_cols if ptypes.is_integer_dtype(train_df_norm[c]) or ptypes.is_float_dtype(train_df_norm[c])]

X_train_raw, train_names = one_hot_sparse(train_df_norm[feat_cols], numeric_cols=numeric_cols)
y_train = train_df_norm["fuel_kg_min"].astype(np.float32).to_numpy()

X_test_raw, test_names = one_hot_sparse(test_df_norm[feat_cols], numeric_cols=numeric_cols)
# Align test to train column space
X_test = align_sparse_to_ref(X_test_raw, test_names, train_names, dtype=np.float32)

# Final DMatrix
dtrain_full = xgb.DMatrix(X_train_raw.astype(np.float32), label=y_train, feature_names=train_names)
log(f"X_train rows: {X_train_raw.shape[0]} | y_train length: {len(y_train)}")

# ----------------------------
# Folds: flight_id groups (+ optional type/duration OOS if present)
# ----------------------------
K = 5
folds_fid = make_group_folds(tr_fids, K=K, rng=rng)

folds_type = None
if "aircraft_type" in all_df.columns:
    tr_types = all_df.loc[~all_df["flight_id"].isin(test_flights), "aircraft_type"].values
    folds_type = make_group_folds(tr_types, K=3, rng=rng)

folds_dur = None
if "flight_duration_min" in all_df.columns:
    tr_dur = all_df.loc[~all_df["flight_id"].isin(test_flights), "flight_duration_min"].astype(float)
    q = np.nanquantile(tr_dur, [0, 0.25, 0.5, 0.75, 1.0])
    # bucketize
    bins = np.unique(q)
    dur_bucket = pd.cut(tr_dur, bins=bins, include_lowest=True)
    folds_dur = make_group_folds(dur_bucket.astype(str).values, K=3, rng=rng)

# ----------------------------
# Validation split (10%)
# ----------------------------
n_tr = X_train_raw.shape[0]
valid_idx = rng.choice(np.arange(n_tr), size=max(1, int(0.10 * n_tr)), replace=False)
train_idx = np.setdiff1d(np.arange(n_tr), valid_idx)

dvalid = xgb.DMatrix(X_train_raw[valid_idx], label=y_train[valid_idx], feature_names=train_names)
dtrain = xgb.DMatrix(X_train_raw[train_idx], label=y_train[train_idx], feature_names=train_names)
watchlist = [(dtrain, "train"), (dvalid, "eval")]

# ----------------------------
# Skip/Load Logic
# ----------------------------
LOAD_MODEL = OUT_MODEL_JSON.exists()
LOAD_METADATA = OUT_META_JSON.exists()

SKIP_BAYESOPT = False
SKIP_FINAL_TRAINING = False

final_params: Dict[str, object] = {}
best_iter_from_meta: Optional[int] = None
X_train_names = train_names

if LOAD_MODEL:
    log(f"⚡ Loading model: {OUT_MODEL_JSON}")
    bst = xgb.Booster()
    bst.load_model(str(OUT_MODEL_JSON))
    final_model = bst
    if LOAD_METADATA:
        meta = json.loads(OUT_META_JSON.read_text(encoding="utf-8"))
        final_params = meta.get("params", {})
        X_train_names = meta.get("feature_names", train_names)
        best_iter_from_meta = int(meta.get("best_iteration", 0) or 0)
    SKIP_BAYESOPT = True
    SKIP_FINAL_TRAINING = True
elif LOAD_METADATA:
    log(f"⚡ Loading metadata: {OUT_META_JSON}")
    meta = json.loads(OUT_META_JSON.read_text(encoding="utf-8"))
    final_params = meta.get("params", {})
    X_train_names = meta.get("feature_names", train_names)
    best_iter_from_meta = int(meta.get("best_iteration", 0) or 0)
    SKIP_BAYESOPT = True
    SKIP_FINAL_TRAINING = False

# ----------------------------
# CV helper
# ----------------------------
def cv_with_timer(params: Dict[str, object],
                  dtrain_cv: xgb.DMatrix,
                  folds: List[np.ndarray],
                  nrounds_cv: int,
                  early_stop: int,
                  label: str) -> Tuple[float, int]:
    import time
    t0 = time.time()
    cv = xgb.cv(
        params=params,
        dtrain=dtrain_cv,
        folds=folds,
        num_boost_round=nrounds_cv,
        early_stopping_rounds=early_stop,
        verbose_eval=False,
        as_pandas=True,
        seed=1337,
    )
    elapsed = time.time() - t0
    # columns like: test-rmse-mean
    test_rmse_mean = cv.filter(like="test-rmse-mean").iloc[:, 0]
    best_iter = int(np.argmin(test_rmse_mean) + 1)  # 1-based rounds
    best_val = float(test_rmse_mean.min())
    log(f"[CV {label}] {elapsed:.1f}s | best={best_val:.4f} | best_iter={best_iter}")
    return best_val, best_iter

# ----------------------------
# BayesOpt (if not skipped and package present)
# ----------------------------
if not SKIP_BAYESOPT and _HAS_BAYESOPT:
    log("🔎 Starting Bayesian Optimization …")

    def scorer(eta, max_depth, min_child_weight, subsample,
               colsample_bytree, gamma, reg_lambda, reg_alpha,
               max_leaves, grow_policy):
        # map to params
        gp = "depthwise" if grow_policy < 0.5 else "lossguide"

        # Prefer GPU, but handle environments without CUDA gracefully
        base_params = gpu_params()
        try_params = base_params.copy()

        try_params.update(dict(
            objective="reg:squarederror",
            eval_metric="rmse",
            nthread=max(2, os.cpu_count() - 1 if os.cpu_count() else 2),
            eta=float(eta),
            min_child_weight=float(min_child_weight),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            gamma=float(gamma),
            reg_lambda=float(reg_lambda),
            reg_alpha=float(reg_alpha),
            grow_policy=gp,
            max_bin=256,
        ))

        if gp == "depthwise":
            try_params["max_depth"] = int(round(max_depth))
            try_params["max_leaves"] = 0
        else:
            try_params["max_depth"] = 0
            try_params["max_leaves"] = max(16, int(round(max_leaves)))

        nrounds_cv = 6000
        early_stop = 60

        try:
            v_fid, best_fid = cv_with_timer(try_params, dtrain_full, folds_fid, nrounds_cv, early_stop, "fid")
        except xgb.core.XGBoostError:
            # fallback to CPU on environments without CUDA
            try_params.update(cpu_params())
            v_fid, best_fid = cv_with_timer(try_params, dtrain_full, folds_fid, nrounds_cv, early_stop, "fid")

        rmse_list = [v_fid]
        best_iter = best_fid

        if folds_type is not None:
            v_type, _ = cv_with_timer(try_params, dtrain_full, folds_type, nrounds_cv, early_stop, "type")
            rmse_list.append(v_type)
        if folds_dur is not None:
            v_dur, _ = cv_with_timer(try_params, dtrain_full, folds_dur, nrounds_cv, early_stop, "dur")
            rmse_list.append(v_dur)

        rmse_mean = float(np.mean(rmse_list))
        rmse_worst = float(np.max(rmse_list))
        # maximize negative loss
        score = -(rmse_mean + 0.5 * rmse_worst)
        # we will read best_iter from logs; return it via a closure? not necessary for BayesOpt
        return score

    pbounds = dict(
        eta=(0.001, 0.7),
        max_depth=(6, 32),
        min_child_weight=(1.0, 256.0),
        subsample=(0.3, 1.0),
        colsample_bytree=(0.3, 1.0),
        gamma=(0.0, 20.0),
        reg_lambda=(0.0, 25.0),
        reg_alpha=(0.0, 20.0),
        max_leaves=(16, 256),
        grow_policy=(0.0, 1.0),
    )

    optimizer = BayesianOptimization(
        f=scorer, pbounds=pbounds, random_state=1337, verbose=2
    )

    # ------ Handle new vs old maximize API ------
    import inspect
    sig = inspect.signature(optimizer.maximize)
    #This defines the amount of initial points Default: both 25
    kwargs = dict(init_points=25, n_iter=50)

    if "acquisition_function" in sig.parameters:
        # New API (v2+ / v3+)
        try:
            from bayes_opt.acquisition import AcquisitionFunction  # type: ignore
            acqf = AcquisitionFunction(kind="ei", xi=0.01, kappa=2.576)
            kwargs["acquisition_function"] = acqf
        except Exception:
            # If import fails, run with defaults (package internal default is EI)
            pass
    elif "acq" in sig.parameters:
        # Old API
        kwargs.update(dict(acq="ei", xi=0.01, kappa=2.576))

    optimizer.maximize(**kwargs)

    best = optimizer.max
    log(f"BayesOpt best: {best}")

    # Extract best params (cast types & grow_policy)
    p = optimizer.max["params"]
    final_gp = "depthwise" if p["grow_policy"] < 0.5 else "lossguide"
    final_params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        nthread=max(2, os.cpu_count() - 1 if os.cpu_count() else 2),
        eta=float(p["eta"]),
        min_child_weight=float(p["min_child_weight"]),
        subsample=float(p["subsample"]),
        colsample_bytree=float(p["colsample_bytree"]),
        gamma=float(p["gamma"]),
        reg_lambda=float(p["reg_lambda"]),
        reg_alpha=float(p["reg_alpha"]),
        grow_policy=final_gp,
        max_bin=256,
    )
    if final_gp == "depthwise":
        final_params["max_depth"] = int(round(p["max_depth"]))
        final_params["max_leaves"] = 0
    else:
        final_params["max_depth"] = 0
        final_params["max_leaves"] = max(16, int(round(p["max_leaves"])))

    # Prefer GPU; if fails, will fallback during training
    final_params.update(gpu_params())

elif not SKIP_BAYESOPT:
    log("⚠️ bayesian-optimization not installed; skipping BayesOpt.")
    SKIP_BAYESOPT = True
    final_params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        nthread=max(2, os.cpu_count() - 1 if os.cpu_count() else 2),
        eta=0.08, min_child_weight=16.0, subsample=0.8,
        colsample_bytree=0.8, gamma=1.0, reg_lambda=3.0, reg_alpha=0.0,
        grow_policy="lossguide", max_depth=0, max_leaves=64, max_bin=256,
        **gpu_params()
    )

# ----------------------------
# Final training (if needed)
# ----------------------------
if not SKIP_FINAL_TRAINING:
    log("🏁 Final training with early stopping …")

    # pick nrounds from BayesOpt history if available (heuristic), else 2000
    nrounds_final = 4000

    # Try GPU training first; fallback to CPU if CUDA not available
    params_for_fit = final_params.copy()
    try:
        final_model = xgb.train(
            params=params_for_fit,
            dtrain=dtrain,
            num_boost_round=nrounds_final,
            evals=watchlist,
            early_stopping_rounds=25,
            verbose_eval=True,
        )
    except xgb.core.XGBoostError:
        params_for_fit.update(cpu_params())
        final_model = xgb.train(
            params=params_for_fit,
            dtrain=dtrain,
            num_boost_round=nrounds_final,
            evals=watchlist,
            early_stopping_rounds=25,
            verbose_eval=True,
        )

    y_valid = y_train[valid_idx]
    bi, best_rmse = get_best_iter_and_valrmse(final_model, dvalid, y_valid)
    log(f"\nBest iteration (final fit): {bi} | eval-RMSE = {best_rmse:.6f}")

else:
    # Loaded model from disk
    #bi = best_iter_from_meta or int(getattr(final_model, "best_iteration", 0) or 0)
    y_valid = y_train[valid_idx]
    bi, best_rmse = get_best_iter_and_valrmse(final_model, dvalid, y_valid)

# ----------------------------
# Evaluation (KG/MIN and per-interval KG)
# ----------------------------
# Validation RMSE (already computed robustly)
val_rmse = best_rmse
val_mean = float(np.nanmean(y_train))
val_rmse_pct = 100.0 * val_rmse / max(val_mean, 1e-6)
log(f"\nValidation RMSE: {val_rmse:.2f} ({val_rmse_pct:.2f}% of mean fuel_kg_min)")

# Test RMSE (kg/min)
X_test_ready = align_sparse_to_ref(X_test, train_names, train_names, dtype=np.float32)
y_test = test_df["fuel_kg_min"].astype(np.float32).to_numpy()
dtest = xgb.DMatrix(X_test_ready, label=y_test, feature_names=train_names)

pred_test = pred_with_best(final_model, dtest, bi)
rmse_test = float(np.sqrt(np.nanmean((pred_test - y_test) ** 2)))
rmse_test_pct = 100.0 * rmse_test / max(float(np.nanmean(y_test)), 1e-6)
log(f"\nTEST RMSE (kg/min): {rmse_test:.2f} ({rmse_test_pct:.2f}% of mean)")

# Per-interval kg
interval_len_test = all_df.loc[all_df["flight_id"].isin(test_flights), "interval_min"].astype(float).to_numpy()
if len(interval_len_test) != len(y_test):
    raise RuntimeError("interval_min length mismatch for test set")

y_test_total = y_test * interval_len_test
pred_test_total = pred_test.astype(np.float32) * interval_len_test.astype(np.float32)

rmse_test_total = float(np.sqrt(np.nanmean((pred_test_total - y_test_total) ** 2)))
rmse_total_pct = 100.0 * rmse_test_total / max(float(np.nanmean(y_test_total)), 1e-6)
log(f"\nTEST RMSE (fuel per interval, single model): {rmse_test_total:.2f} kg ({rmse_total_pct:.2f}% of mean interval fuel)")

# ----------------------------
# Ensemble over seeds
# ----------------------------
M = 10
seeds = list(range(2001, 2001 + M))
pred_mat = np.empty((len(y_test), M), dtype=np.float32)

for j, sd in enumerate(seeds, start=1):
    log(f"\nEnsemble Model {j} / {M}")
    params_j = final_params.copy()
    # slight shrinkage
    params_j["subsample"] = min(0.9, float(params_j.get("subsample", 0.9)))
    params_j["colsample_bytree"] = min(0.9, float(params_j.get("colsample_bytree", 0.9)))
    params_j["seed"] = int(sd)

    try:
        bst_j = xgb.train(
            params=params_j,
            dtrain=dtrain,
            num_boost_round=2000,
            evals=watchlist,
            early_stopping_rounds=50,
            verbose_eval=False,
        )
    except xgb.core.XGBoostError:
        # fallback to CPU
        params_j.update(cpu_params())
        bst_j = xgb.train(
            params=params_j,
            dtrain=dtrain,
            num_boost_round=2000,
            evals=watchlist,
            early_stopping_rounds=50,
            verbose_eval=False,
        )

    bi_j = int(getattr(bst_j, "best_iteration", 0) or 0)
    pred_mat[:, j - 1] = pred_with_best(bst_j, dtest, bi_j)

pred_ens = pred_mat.mean(axis=1)
rmse_ens = float(np.sqrt(np.nanmean((pred_ens - y_test) ** 2)))
rmse_ens_pct = 100.0 * rmse_ens / max(float(np.nanmean(y_test)), 1e-6)
log(f"\nEnsemble TEST RMSE (kg/min): {rmse_ens:.3f} ({rmse_ens_pct:.2f}% of mean)")

# Ensemble per-interval
pred_total = pred_ens.astype(np.float32) * interval_len_test.astype(np.float32)
rmse_total = float(np.sqrt(np.nanmean((pred_total - y_test_total) ** 2)))
rmse_total_pct = 100.0 * rmse_total / max(float(np.nanmean(y_test_total)), 1e-6)
log(f"Ensemble TEST RMSE (interval kg): {rmse_total:.3f} ({rmse_total_pct:.2f}% of mean)")

# ----------------------------
# Importance & Save
# ----------------------------
# Importance from native booster (gain)
im_dict = final_model.get_score(importance_type="gain")
imp_df = pd.DataFrame(
    {"feature": list(im_dict.keys()), "importance": list(im_dict.values())}
).sort_values("importance", ascending=False)
imp_df.to_csv(OUT_IMPORTANCE_CSV, index=False)
log(f"Saved importance to {OUT_IMPORTANCE_CSV}")

# Save model and metadata
final_model.save_model(str(OUT_MODEL_JSON))
meta_payload = dict(
    feature_names=train_names,
    params=final_params,
    best_iteration=bi,
    test_rmse_interval_kg=rmse_total,
    test_rmse_kg_min=rmse_ens,
)
OUT_META_JSON.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
log(f"Saved model -> {OUT_MODEL_JSON}\nSaved meta  -> {OUT_META_JSON}")

# ----------------------------
# Full error table (test rows)
# ----------------------------
test_rows_full = all_df[all_df["flight_id"].isin(test_flights)].copy()
actual_kg = y_test_total
predicted_kg = pred_total
abs_error = np.abs(predicted_kg - actual_kg)
pct_error = 100.0 * abs_error / np.maximum(actual_kg, 1e-6)

pred_df = pd.DataFrame({
    "flight_id": te_fids,
    "actual_kg": actual_kg,
    "predicted_kg": predicted_kg,
    "abs_error": abs_error,
    "pct_error": pct_error
})
# Join with original columns for richer feature set
df_full = pd.concat([test_rows_full.reset_index(drop=True), pred_df[["actual_kg","predicted_kg","abs_error","pct_error"]]], axis=1)
df_full = df_full.sort_values("abs_error", ascending=False)
df_full.to_csv(OUT_PRED_ERR_CSV, index=False)
log(f"\n✅ Pred vs Actual table: {OUT_PRED_ERR_CSV} ({df_full.shape[0]} rows, {df_full.shape[1]} cols)")
log(df_full[["idx", "flight_id", "actual_kg", "predicted_kg", "abs_error", "pct_error"]].head(10).to_string(index=False))

# ----------------------------
# Submission (with taxi preds)
# ----------------------------
if not IN_SUBMISSION.exists():
    raise FileNotFoundError(f"Missing {IN_SUBMISSION}")

sub_df = pd.read_csv(IN_SUBMISSION)
sub_df = sub_df.assign(
    status=lambda d: np.where(d["pct_elapsed_mid"] < 0, "taxi_out",
             np.where(d["pct_elapsed_mid"] > 100, "taxi_in", "inflight"))
)
log(f"Rows in submission: {len(sub_df)}")

only_taxi = sub_df[sub_df["status"].isin(["taxi_out","taxi_in"])][["idx","flight_id","start","end"]].copy()
sub_df = sub_df[sub_df["status"] == "inflight"].copy()

# Taxi predictions (already scored)
if not IN_TAXI_SCORED.exists():
    raise FileNotFoundError(f"Missing {IN_TAXI_SCORED}")
taxi_pred = (
    pd.read_csv(IN_TAXI_SCORED)
    .drop(columns=[c for c in ["fuel_kg_min", "fuel_kg"] if c in ["fuel_kg_min","fuel_kg"]], errors="ignore")
    .rename(columns={"fuel_kg_pred":"fuel_kg"})
    [["idx", "fuel_kg"]]
    .merge(only_taxi, on="idx", how="left")
    [["idx","flight_id","start","end","fuel_kg"]]
)

# Build the same feature processing for submission
feat_cols_sub = [c for c in sub_df.columns if c != "fuel_kg_min"]
sub_norm = normalize_columns_for_model(sub_df, feat_cols_sub)

# Ensure all vars in training design exist
missing_vars = [c for c in train_df_norm[feat_cols].columns if c not in sub_norm.columns]
for mv in missing_vars:
    sub_norm[mv] = np.nan

X_sub_raw, sub_names = one_hot_sparse(sub_norm[feat_cols], numeric_cols=numeric_cols)
X_sub = align_sparse_to_ref(X_sub_raw, sub_names, train_names, dtype=np.float32)
dsub = xgb.DMatrix(X_sub, feature_names=train_names)

pred_sub_min = pred_with_best(final_model, dsub, bi)
sub_df = sub_df.copy()
if "interval_min" not in sub_df.columns:
    raise ValueError("submission_intervals.csv must contain `interval_min`")
sub_df["fuel_kg"] = pred_sub_min * sub_df["interval_min"].astype(float)

qs = np.quantile(pred_sub_min, [0, 0.01, 0.5, 0.99, 1.0])
log("pred_sub_min quantiles: " + " | ".join(f"{q:.3f}" for q in qs))

sub_out = sub_df[["idx","flight_id","start","end","fuel_kg"]]
sub_out = pd.concat([sub_out, taxi_pred], axis=0, ignore_index=True)

# Write parquet
if not _HAS_ARROW:
    raise RuntimeError("pyarrow not installed. Please `pip install pyarrow` to write parquet.")
table = pa.Table.from_pandas(sub_out)
pq.write_table(table, OUT_PARQUET)
log(f"✅ Parquet written: {OUT_PARQUET} | rows: {len(sub_out)}")

# ----------------------------
# Upload to OpenSky S3 (MinIO client `mc`)
# ----------------------------
MC = "mc.exe" if os.name == "nt" else "mc"
local_file = str(OUT_PARQUET)
target_path = "opensky/prc-2025-honest-rose/honest-rose_v44.parquet"

log("Configuring mc alias 'opensky' …")
alias_cmd = [
    "alias", "set", "--api", "S3v4", "--path", "auto",
    "opensky", "https://s3.opensky-network.org",
    "3tdiGZNiuaKj9I7S", "tb1RouZ1LHRYU3ZUIMy5TFGzj4sSYgTB"
]
try:
    subprocess.run([MC] + alias_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
except Exception as e:
    log(f"⚠️ MinIO alias set failed (continuing): {e}")

log(f"Uploading: {local_file} → {target_path}")
try:
    subprocess.run([MC, "cp", local_file, target_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log("✅ Upload command executed; verifying …")
    ls = subprocess.run([MC, "ls", "opensky/prc-2025-honest-rose/"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if "honest-rose_v44.parquet" in ls.stdout:
        log("🎉 File visible in bucket!")
    else:
        log("⚠️ Upload executed, but file not listed — possibly cache delay.")
except Exception as e:
    log(f"❌ Upload failed:\n{e}")
