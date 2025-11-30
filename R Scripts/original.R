„# ===========================
# Fuel burn – XGBoost + BayesOpt + Submission + Upload
# Version: ohne setTimeLimit, ohne Residual-Modelle
# ===========================

required_pkgs <- c(
  "data.table","Matrix","ParBayesianOptimization",
  "Metrics","doParallel","foreach","backports",
  "arrow","tidyverse"
)
to_install <- setdiff(required_pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)
invisible(lapply(required_pkgs, function(p) suppressPackageStartupMessages(library(p, character.only = TRUE))))

library(xgboost)

`%||%` <- function(x, y) if (!is.null(x)) x else y

# =========================================================
# Helper: robust predict with best iteration
# =========================================================
pred_with_best <- function(model, dmat, best_iter = NULL) {
  nt <- tryCatch(xgboost::xgb.num_trees(model), error = function(e) NA_integer_)
  cand <- c(
    suppressWarnings(as.integer(best_iter)),
    suppressWarnings(as.integer(model$best_ntreelimit)),
    suppressWarnings(as.integer(model$best_iteration)),
    suppressWarnings(as.integer(nt))
  )
  bi <- cand[which(is.finite(cand) & !is.na(cand) & cand > 0L)][1]
  if (!length(bi)) bi <- 1L
  if (is.finite(nt) && !is.na(nt) && nt > 0L) {
    bi <- max(1L, min(bi, nt))
  } else {
    bi <- max(1L, bi)
  }
  predict(model, dmat, ntreelimit = bi)
}

cv_with_timer <- function(params, data, folds, nrounds_cv, early_stop, label="cv") {
  ptm <- proc.time()
  cv <- xgboost::xgb.cv(
    params = params, data = data,
    nrounds = nrounds_cv, folds = folds,
    early_stopping_rounds = early_stop, verbose = 0
  )
  elapsed <- (proc.time() - ptm)[["elapsed"]]
  cat(sprintf("[CV %s] %.1fs | best=%.4f | best_iter=%d\n",
              label, elapsed,
              min(cv$evaluation_log$test_rmse_mean), cv$best_iteration))
  cv
}

normalize_for_mm <- function(DT, feat_cols) {
  for (col in feat_cols) {
    v <- DT[[col]]
    if (is.logical(v)) {
      v <- as.integer(v); v[is.na(v)] <- 0L
    } else if (is.numeric(v)) {
      v[!is.finite(v)] <- NA_real_
    } else {
      v <- as.character(v)
      v[is.na(v)] <- "__NA__"
      v <- factor(v)
    }
    DT[[col]] <- v
  }
  DT
}

align_to_train <- function(X, refnames) {
  miss <- setdiff(refnames, colnames(X))
  if (length(miss)) {
    X <- Matrix::cbind2(
      X,
      Matrix::Matrix(0, nrow(X), length(miss), sparse = TRUE,
                     dimnames = list(NULL, miss))
    )
  }
  X <- X[, refnames, drop = FALSE]
  X
}

set.seed(1337)

# =========================================================
# Load & basic filtering
# =========================================================
csv_path <- "C:/PRC Data Challenge 2025/features_intervals.csv"
stopifnot(file.exists(csv_path))

all_data <- read.csv(csv_path, stringsAsFactors = FALSE) |>
  dplyr::mutate(
    status = dplyr::case_when(
      pct_elapsed_mid < 0   ~ "taxi_out",
      pct_elapsed_mid > 100 ~ "taxi_in",
      TRUE                  ~ "inflight"
    )
  ) |>
  dplyr::filter(!(status %in% c("taxi_in","taxi_out")))

dt <- data.table::as.data.table(all_data)
stopifnot("fuel_kg_min" %in% names(dt))
dt <- dt[!is.na(fuel_kg_min)]

# Drop komplett konstante oder komplett NA-Spalten
drop_cols_const <- names(dt)[vapply(dt, function(x) all(is.na(x)) || length(unique(x)) <= 1, logical(1))]
if (length(drop_cols_const)) dt[, (drop_cols_const) := NULL]

# =========================================================
# Split by flight_id, Feature-Set definieren
# =========================================================
dt_grp <- data.table::copy(dt)

drop_id_cols <- c(
  "idx","flight_id","start","end","flight_date","takeoff","landed",
  "start_hour_utc","end_hour_utc","midpoint_utc","model_time_utc",
  "points_file_exists","origin_icao","dest_icao","dow","month",
  "weather_code_text","precipitation","origin_region","dest_region",
  "status"
)

keep_cols <- setdiff(names(dt_grp), drop_id_cols)
stopifnot("fuel_kg_min" %in% keep_cols)

all_flights  <- unique(dt_grp$flight_id)
test_flights <- sample(all_flights, size = floor(0.20 * length(all_flights)))

train_dt <- dt_grp[!(flight_id %in% test_flights), ..keep_cols]
test_dt  <- dt_grp[ (flight_id %in% test_flights), ..keep_cols]

tr_fid <- dt_grp[!(flight_id %in% test_flights), flight_id]

# =========================================================
# Sparse Matrices
# =========================================================
library(Matrix)

tr <- data.table::copy(train_dt)
te <- data.table::copy(test_dt)

stopifnot("fuel_kg_min" %in% names(tr))
feat_cols <- setdiff(names(tr), "fuel_kg_min")

tr <- normalize_for_mm(tr, feat_cols)
te <- normalize_for_mm(te, feat_cols)

tt <- as.formula(paste("~ 0 +", paste(feat_cols, collapse = " + ")))
options(na.action = "na.pass")

X_train <- Matrix::sparse.model.matrix(tt, data = tr, na.action = stats::na.pass)
X_test  <- Matrix::sparse.model.matrix(tt, data = te, na.action = stats::na.pass)

y_train <- as.numeric(tr$fuel_kg_min)
y_test  <- as.numeric(te$fuel_kg_min)

cat("X_train rows:", nrow(X_train), "| y_train length:", length(y_train), "\n")
stopifnot(nrow(X_train) == length(y_train))

dtrain_full <- xgboost::xgb.DMatrix(data = X_train, label = y_train)
dtest       <- xgboost::xgb.DMatrix(data = X_test,  label = y_test)

# =========================================================
# Folds (flight_id + optional type/duration OOS)
# =========================================================
K <- 5
groups <- unique(tr_fid)
fold_assign <- sample(rep(1:K, length.out = length(groups)))
folds <- lapply(1:K, function(k) {
  gk <- groups[fold_assign == k]
  which(tr_fid %in% gk)
})
folds_fid <- folds

make_group_folds <- function(groups, K=3) {
  ug <- unique(groups)
  assign <- sample(rep(1:K, length.out=length(ug)))
  lapply(1:K, function(k) which(groups %in% ug[assign==k]))
}

# Type-OOS
if ("aircraft_type" %in% names(dt_grp)) {
  tr_types <- dt_grp[!(flight_id %in% test_flights), aircraft_type]
  folds_type <- make_group_folds(tr_types, K=3)
} else {
  folds_type <- NULL
}

# Duration-OOS
if ("flight_duration_min" %in% names(dt_grp)) {
  tr_dur <- dt_grp[!(flight_id %in% test_flights), flight_duration_min]
  q <- quantile(tr_dur, probs = c(0,.25,.5,.75,1), na.rm=TRUE)
  dur_bucket <- cut(tr_dur, breaks = unique(q), include.lowest = TRUE)
  folds_dur <- make_group_folds(dur_bucket, K=3)
} else {
  folds_dur <- NULL
}

# =========================================================
# Validation-Split für Final-Training
# =========================================================
n_tr <- nrow(X_train)
valid_idx <- sample.int(n_tr, size = max(1L, floor(0.10 * n_tr)))
dvalid <- xgboost::xgb.DMatrix(X_train[valid_idx, ],  label = y_train[valid_idx])
dtrain <- xgboost::xgb.DMatrix(X_train[-valid_idx, ], label = y_train[-valid_idx])
watchlist <- list(train = dtrain, eval = dvalid)

# =========================================================
# Skip/Load-Logik
# =========================================================
LOAD_MODEL    <- TRUE   # fertiges Modell laden?
LOAD_METADATA <- FALSE  # nur Meta laden?

model_path <- "C:/PRC Data Challenge 2025/xgb_fuel_burn_final_model.rds"
meta_path  <- "C:/PRC Data Challenge 2025/xgb_fuel_burn_metadata.rds"

SKIP_BAYESOPT      <- FALSE
SKIP_FINAL_TRAINING <- FALSE

if (LOAD_MODEL && file.exists(model_path)) {
  message("⚡ Lade fertiges Modell: ", model_path)
  final_model <- readRDS(model_path)
  if (file.exists(meta_path)) {
    meta <- readRDS(meta_path)
    final_params   <- meta$params
    X_train_names  <- meta$feature_names
    best_iter      <- meta$best_iteration
  } else {
    final_params   <- final_model$params
    X_train_names  <- final_model$feature_names
    best_iter      <- final_model$best_iteration
  }
  SKIP_BAYESOPT <- TRUE
  SKIP_FINAL_TRAINING <- TRUE
}

if (!LOAD_MODEL && LOAD_METADATA && file.exists(meta_path)) {
  message("⚡ Lade Metadaten (beste Hyperparameter): ", meta_path)
  meta <- readRDS(meta_path)
  final_params  <- meta$params
  X_train_names <- meta$feature_names
  best_iter     <- meta$best_iteration
  
  if (is.numeric(final_params$grow_policy))
    final_params$grow_policy <- if (final_params$grow_policy < 0.5) "depthwise" else "lossguide"
  
  SKIP_BAYESOPT      <- TRUE
  SKIP_FINAL_TRAINING <- FALSE
}

if (!LOAD_MODEL && !LOAD_METADATA) {
  SKIP_BAYESOPT      <- FALSE
  SKIP_FINAL_TRAINING <- FALSE
}

# =========================================================
# BayesOpt (nur wenn nötig) – ohne setTimeLimit!
# =========================================================
if (!SKIP_BAYESOPT) {
  
  scorer <- function(eta, max_depth, min_child_weight, subsample,
                     colsample_bytree, gamma, lambda, alpha,
                     max_leaves, grow_policy) {
    nrounds_cv <- 6000L; early_stop <- 60L
    
    gp <- if (grow_policy < 0.5) "depthwise" else "lossguide"
    params <- list(
      objective="reg:squarederror", eval_metric="rmse",
      device="cuda", tree_method="hist",
      nthread=min(6L, max(2L, parallel::detectCores()-1L)),
      single_precision_histogram=1, max_bin=256,
      eta=eta, min_child_weight=min_child_weight, subsample=subsample,
      colsample_bytree=colsample_bytree, gamma=gamma, lambda=lambda,
      alpha=alpha, grow_policy=gp
    )
    if (gp=="depthwise") {
      params$max_depth  <- as.integer(round(max_depth))
      params$max_leaves <- 0L
    } else {
      params$max_depth  <- 0L
      params$max_leaves <- max(16L, as.integer(round(max_leaves)))
    }
    
    rmse_list <- c(); best_iter <- 500L
    
    out <- try({
      cv_fid <- cv_with_timer(params, dtrain_full, folds_fid, nrounds_cv, early_stop, "fid"); gc()
      rmse_list <- c(rmse_list, min(cv_fid$evaluation_log$test_rmse_mean))
      best_iter <- cv_fid$best_iteration
      
      if (!is.null(folds_type)) {
        cv_type <- cv_with_timer(params, dtrain_full, folds_type, nrounds_cv, early_stop, "type"); gc()
        rmse_list <- c(rmse_list, min(cv_type$evaluation_log$test_rmse_mean))
      }
      if (!is.null(folds_dur)) {
        cv_dur <- cv_with_timer(params, dtrain_full, folds_dur, nrounds_cv, early_stop, "dur"); gc()
        rmse_list <- c(rmse_list, min(cv_dur$evaluation_log$test_rmse_mean))
      }
      
      TRUE
    }, silent = TRUE)
    
    if (!isTRUE(out)) {
      cat("⚠️ scorer timeout/err → penalize point\n")
      return(list(Score = -1e9, nrounds = best_iter))
    }
    
    rmse_mean  <- mean(rmse_list)
    rmse_worst <- max(rmse_list)
    list(Score = -(rmse_mean + 0.5 * rmse_worst), nrounds = best_iter)
  }
  
  bounds <- list(
    eta               = c(0.01, 0.5),
    max_depth         = c(8L, 32L),
    min_child_weight  = c(4, 128),
    subsample         = c(0.4, 1.0),
    colsample_bytree  = c(0.4, 1.0),
    gamma             = c(0.0, 15.0),
    lambda            = c(0.0, 15.0),
    alpha             = c(0.0, 8.0),
    max_leaves        = c(32L, 128L),
    grow_policy       = c(0, 1.0)
  )
  
  opt <- ParBayesianOptimization::bayesOpt(
    FUN = scorer, bounds = bounds,
    initPoints = 25, iters.n = 25,
    acq = "ei", parallel = FALSE,
    gsPoints = 200L, plotProgress = FALSE, verbose = 1
  )
  best <- ParBayesianOptimization::getBestPars(opt)
  print(best)
  
  final_grow_policy <- if (best$grow_policy < 0.5) "depthwise" else "lossguide"
  final_params <- list(
    objective="reg:squarederror", eval_metric="rmse",
    device="cuda", tree_method="hist",
    nthread=min(6L, max(2L, parallel::detectCores()-1L)),
    single_precision_histogram=1, max_bin=256,
    eta=best$eta, min_child_weight=best$min_child_weight,
    subsample=best$subsample, colsample_bytree=best$colsample_bytree,
    gamma=best$gamma, lambda=best$lambda, alpha=best$alpha,
    grow_policy=final_grow_policy
  )
  if (final_grow_policy=="depthwise") {
    final_params$max_depth  <- as.integer(round(best$max_depth))
    final_params$max_leaves <- 0L
  } else {
    final_params$max_depth  <- 0L
    final_params$max_leaves <- max(16L, as.integer(round(best$max_leaves)))
  }
}

# Safety: Meta geladen, aber kein opt
if (LOAD_METADATA && exists("final_params") && !exists("opt")) {
  message("ℹ️ Loaded metadata, proceeding to final training.")
} else if (LOAD_METADATA && !exists("final_params")) {
  stop("❌ Keine final_params in Metadaten gefunden – überprüfe ", meta_path)
}

# =========================================================
# Final Training (falls nötig)
# =========================================================
if (!SKIP_FINAL_TRAINING) {
  message("🏁 Starte Final-Training mit Early Stopping …")
  
  if (exists("opt") && !is.null(opt$scoreSummary)) {
    ss <- opt$scoreSummary
    nbest <- ss$nrounds[which.max(ss$Value)]
    nrounds_final <- max(500L, as.integer(round(nbest)) + 400L)
  } else {
    nrounds_final <- 2000L
  }
  
  final_model <- xgboost::xgb.train(
    params = final_params,
    data   = dtrain,
    nrounds = nrounds_final,
    watchlist = watchlist,
    early_stopping_rounds = 25,
    verbose = 1
  )
  
  cat(sprintf("\nBest iteration (final fit): %d | eval-RMSE = %.6f\n",
              final_model$best_iteration,
              final_model$evaluation_log[final_model$best_iteration,]$eval_rmse))
}

# =========================================================
# Evaluation (Test RMSE, kg/min & kg/interval, Ensemble)
# =========================================================
if (!exists("X_train_names")) X_train_names <- colnames(X_train)
if (!exists("best_iter") || is.null(best_iter)) best_iter <- final_model$best_iteration

# Validation RMSE
val_rmse <- final_model$evaluation_log[final_model$best_iteration,]$eval_rmse
val_mean <- mean(y_train)
val_rmse_pct <- (val_rmse / pmax(val_mean, 1e-6)) * 100
cat(sprintf("\nValidation RMSE: %.2f (%.2f%% of mean fuel_kg_min)\n",
            val_rmse, val_rmse_pct))

# Test-RMSE (kg/min)
X_test <- Matrix::sparse.model.matrix(tt, data = te, na.action = stats::na.pass)
X_test <- align_to_train(X_test, colnames(X_train))
cat("X_test rows:", nrow(X_test), "| y_test length:", length(y_test), "\n")
stopifnot(nrow(X_test) == length(y_test))

dtest <- xgboost::xgb.DMatrix(X_test, label = y_test)
pred_test <- pred_with_best(final_model, dtest, final_model$best_iteration)

rmse_test <- Metrics::rmse(y_test, pred_test)
eps_mean  <- max(mean(y_test, na.rm = TRUE), 1e-6)
rmse_test_pct <- (rmse_test / eps_mean) * 100
cat(sprintf("\nTEST RMSE (kg/min): %.2f (%.2f%% of mean)\n",
            rmse_test, rmse_test_pct))

# Test-RMSE auf kg / Intervall
test_rows_dtgrp <- dt_grp[(flight_id %in% test_flights)]
interval_len_test <- as.numeric(test_rows_dtgrp$interval_min)
stopifnot(length(interval_len_test) == length(y_test))

y_test_total    <- y_test * interval_len_test
pred_test_total <- as.numeric(pred_test) * interval_len_test

rmse_test_total <- Metrics::rmse(y_test_total, pred_test_total)
rmse_total_pct  <- (rmse_test_total / pmax(mean(y_test_total, na.rm = TRUE), 1e-6)) * 100

cat(sprintf("\nTEST RMSE (fuel per interval, single model): %.2f kg (%.2f%% of mean interval fuel)\n",
            rmse_test_total, rmse_total_pct))

# Ensemble über Seeds
M <- 10L
seeds <- 2001:(2000+M)
pred_mat <- matrix(NA_real_, nrow = length(y_test), ncol = M)
best_iters <- integer(M)

for (j in seq_len(M)) {
  cat(sprintf("\nEnsemble Model %d / %d\n", j, M))
  set.seed(seeds[j])
  
  params_j <- modifyList(final_params, list(
    subsample = min(0.9, final_params$subsample),
    colsample_bytree = min(0.9, final_params$colsample_bytree)
  ))
  
  bst_j <- xgboost::xgb.train(
    params   = params_j,
    data     = dtrain,
    nrounds  = nrounds_final,
    watchlist = watchlist,
    early_stopping_rounds = 50,
    verbose = 0
  )
  
  best_iters[j] <- bst_j$best_ntreelimit %||%
    bst_j$best_iteration %||%
    xgboost::xgb.num_trees(bst_j)
  
  pred_mat[, j] <- pred_with_best(bst_j, dtest, best_iters[j])
  gc()
}

pred_ens <- rowMeans(pred_mat)
rmse_ens <- Metrics::rmse(y_test, pred_ens)
rmse_ens_pct <- 100 * rmse_ens / pmax(mean(y_test), 1e-6)
cat(sprintf("\nEnsemble TEST RMSE (kg/min): %.3f (%.2f%% of mean)\n",
            rmse_ens, rmse_ens_pct))

# Ensemble auf kg / Intervall
interval_len_test <- dt_grp[(flight_id %in% test_flights), interval_min]
stopifnot(length(interval_len_test) == length(y_test))

y_total    <- y_test * interval_len_test
pred_total <- as.numeric(pred_ens) * interval_len_test

ok <- is.finite(y_total) & is.finite(pred_total)
rmse_total     <- sqrt(mean( (pred_total[ok] - y_total[ok])^2 ))
rmse_total_pct <- 100 * rmse_total / pmax(mean(y_total[ok]), 1e-6)

cat(sprintf("Ensemble TEST RMSE (interval kg): %.3f (%.2f%% of mean)\n",
            rmse_total, rmse_total_pct))

# Importance & Save
imp <- xgboost::xgb.importance(model = final_model, feature_names = colnames(X_train))
print(utils::head(imp, 20))
write.csv(imp, "C:/PRC Data Challenge 2025/importance_matrix.csv", row.names = FALSE)

saveRDS(final_model, file = "C:/PRC Data Challenge 2025/xgb_fuel_burn_final_model.rds")
saveRDS(list(
  feature_names = colnames(X_train),
  params = final_params,
  best_iteration = final_model$best_iteration,
  test_rmse = rmse_total
), file = "C:/PRC Data Challenge 2025/xgb_fuel_burn_metadata.rds")

# Vollständiges Fehler-Feature-Set
stopifnot(length(y_test) == nrow(test_dt))
interval_len_test <- dt_grp[(flight_id %in% test_flights), interval_min]
stopifnot(length(interval_len_test) == nrow(test_dt))

actual_kg    <- y_test * interval_len_test
predicted_kg <- pred_ens * interval_len_test
abs_error <- abs(predicted_kg - actual_kg)
pct_error <- 100 * abs_error / pmax(actual_kg, 1e-6)

pred_df <- data.table::data.table(
  flight_id    = dt_grp[(flight_id %in% test_flights), flight_id],
  actual_kg    = actual_kg,
  predicted_kg = predicted_kg,
  abs_error    = abs_error,
  pct_error    = pct_error
)

dt_test_full <- dt_grp[(flight_id %in% test_flights)]
stopifnot(nrow(dt_test_full) == nrow(pred_df))

df_full <- cbind(dt_test_full, pred_df[, .(actual_kg, predicted_kg, abs_error, pct_error)])
df_full <- df_full[order(-abs_error)]

out_err_path <- "C:/PRC Data Challenge 2025/predicted_vs_actual_full.csv"
data.table::fwrite(df_full, out_err_path)
cat(sprintf(
  "\n✅ Vollständiges Fehler-Feature-Set gespeichert unter:\n%s\n(%d Zeilen, %d Spalten)\n",
  out_err_path, nrow(df_full), ncol(df_full)
))
print(head(df_full[, .(idx, flight_id, actual_kg, predicted_kg, abs_error, pct_error)], 10))

# =========================================================
# Submission (mit Taxi-Preds)
# =========================================================
X_train_names <- final_model$feature_names
final_params  <- final_model$params
best_iter     <- final_model$best_iteration

sub_path_in  <- "C:/PRC Data Challenge 2025/submission_intervals.csv"
stopifnot(file.exists(sub_path_in))

submission_df <- data.table::fread(sub_path_in) |>
  dplyr::mutate(
    status = dplyr::case_when(
      pct_elapsed_mid < 0   ~ "taxi_out",
      pct_elapsed_mid > 100 ~ "taxi_in",
      TRUE                  ~ "inflight"
    )
  )

cat("Rows in submission:", nrow(submission_df), "\n")

only_taxi <- submission_df |>
  dplyr::filter(status %in% c("taxi_out","taxi_in")) |>
  dplyr::select(idx, flight_id, start, end)

submission_df <- submission_df |>
  dplyr::filter(status == "inflight")

taxi_pred <- read.csv("C:/PRC Data Challenge 2025/submission_intervals_v4_scored.csv") |>
  dplyr::select(-fuel_kg_min, -fuel_kg) |>
  dplyr::rename(fuel_kg = fuel_kg_pred) |>
  dplyr::select(idx, fuel_kg) |>
  dplyr::left_join(only_taxi, by = "idx") |>
  dplyr::select(idx, flight_id, start, end, fuel_kg)

# gleiche Normalisierung wie Training
feat_cols_sub <- setdiff(colnames(submission_df), c("fuel_kg_min"))
sub_for_mm <- normalize_for_mm(data.table::copy(submission_df), feat_cols_sub)

# alle Variablen im tt sicherstellen
need_vars <- setdiff(all.vars(tt), colnames(sub_for_mm))
if (length(need_vars)) {
  for (v in need_vars) sub_for_mm[[v]] <- NA
}

X_sub <- Matrix::sparse.model.matrix(tt, data = sub_for_mm, na.action = stats::na.pass)
X_sub <- align_to_train(X_sub, X_train_names)
stopifnot(identical(colnames(X_train), colnames(X_sub)))
dsub  <- xgboost::xgb.DMatrix(X_sub)

pred_sub_min <- pred_with_best(final_model, dsub, final_model$best_iteration)

submission_df$fuel_kg <- pred_sub_min * submission_df$interval_min
stopifnot(nrow(submission_df) == nrow(X_sub))
stopifnot(!any(is.na(submission_df$fuel_kg)))

qs <- quantile(pred_sub_min, c(0,.01,.5,.99,1), na.rm=TRUE)
cat("pred_sub_min quantiles:", paste(round(qs,3), collapse=" | "), "\n")

submission_df <- submission_df |>
  dplyr::select(idx, flight_id, start, end, fuel_kg)

submission_df <- dplyr::bind_rows(submission_df, taxi_pred)

out_df <- submission_df[, c("idx","flight_id","start","end","fuel_kg")]
out_path <- "C:/PRC Data Challenge 2025/honest-rose_v20.parquet"

arrow::write_parquet(out_df, out_path)
cat(sprintf("✅ Parquet geschrieben: %s | Zeilen: %d\n", out_path, nrow(out_df)))

# =========================================================
# Upload zu OpenSky S3 (MinIO)
# =========================================================
mc_bin <- "mc.exe"
if (.Platform$OS.type != "windows") mc_bin <- "mc"

local_file  <- "C:/PRC Data Challenge 2025/honest-rose_v20.parquet"
target_path <- "opensky/prc-2025-honest-rose/honest-rose_v20.parquet"

cat("Richte mc alias 'opensky' ein...\n")
alias_cmd <- c("alias", "set", "--api", "S3v4", "--path", "auto",
               "opensky", "https://s3.opensky-network.org",
               "3tdiGZNiuaKj9I7S", "tb1RouZ1LHRYU3ZUIMy5TFGzj4sSYgTB")
system2(mc_bin, args = alias_cmd, stdout = TRUE, stderr = TRUE)

cat(sprintf("Lade hoch: %s → %s\n", local_file, target_path))
res_cp <- tryCatch(
  system2(mc_bin, args = c("cp", shQuote(local_file), shQuote(target_path)),
          stdout = TRUE, stderr = TRUE),
  error = function(e) e
)

if (inherits(res_cp, "error")) {
  cat("❌ Upload FEHLGESCHLAGEN:\n")
  print(res_cp)
} else {
  cat("✅ Upload ausgeführt, prüfe Sichtbarkeit...\n")
  res_ls <- tryCatch(
    system2(mc_bin, args = c("ls", "opensky/prc-2025-honest-rose/"), stdout = TRUE),
    error = function(e) e
  )
  if (inherits(res_ls, "error")) {
    cat("⚠️ Verifikation fehlgeschlagen:\n")
    print(res_ls)
  } else {
    if (any(grepl("honest-rose_v20.parquet", res_ls))) {
      cat("🎉 Datei erfolgreich im Bucket sichtbar!\n")
    } else {
      cat("⚠️ Upload ausgeführt, aber Datei nicht gelistet — evtl. Cache-Verzögerung.\n")
    }
  }
}“
