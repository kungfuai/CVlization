# Analytical Examples Plan

## Tabular

- analytical/tabular/automl/autogluon_structured: AutoGluon end-to-end tabular AutoML (classification/regression) leveraging official Docker images.
- analytical/tabular/automl/pycaret_structured: Low-code AutoML flows with PyCaret’s slim/full containers.
- analytical/tabular/customer_analytics/gbt_telco_churn: Churn prediction use case built on the gbt (LightGBM) wrapper with Telco Customer dataset.
- analytical/tabular/risk_scoring/gbt_credit_default: Credit default modeling using gbt with imbalance handling and explainability.
- analytical/tabular/marketing/gbt_upsell_propensity: Campaign response propensity modeling with uplift-aware metrics using gbt.
- analytical/tabular/uncertainty/quantile_lightgbm: Pinball-loss LightGBM quantile regression delivering calibrated prediction intervals (e.g., California Housing).
- analytical/tabular/uncertainty/catboost_quantile: CatBoost quantile boosting pipeline for asymmetric interval estimates on medium-sized tabular datasets.
- analytical/tabular/uncertainty/mapie_conformal: Model-agnostic conformal prediction intervals powered by MAPIE wrapping any base regressor/classifier.
- analytical/tabular/regression/gbt_housing_prices: Structured regression workflow predicting housing prices with gbt, featuring feature importance, calibration, and scenario analysis.
- analytical/tabular/anomaly_detection/pyod_fraud_detection: Transaction fraud detection pipeline using PyOD ensembles with Dockerized ingestion of public fraud datasets.
- analytical/tabular/survival/pycox_retention: Customer survival / churn-time modeling using PyCox (DeepSurv-style) with evaluation of retention and lifetime metrics.
- analytical/tabular/recommendation/lightgbm_ranking: Learning-to-rank or implicit recommendation baseline using LightGBM ranker on public ranking datasets (e.g., Yahoo! LTR).
- analytical/tabular/feature_engineering/autofe_structured: Automated feature engineering pipeline combining Featuretools / autofeat with downstream gbt baseline for tabular modeling.
- analytical/tabular/uncertainty/conformal_lightgbm: Conformal prediction intervals and post-hoc calibration on LightGBM (gbt) outputs, covering both regression quantiles and classification probability calibration (Platt/Isotonic).
- analytical/tabular/uncertainty/pymc_bayesian_regression: Bayesian regression via PyMC for interpretable posterior inference and credible intervals on medium-sized structured datasets.
- analytical/tabular/causal/dowhy_policy_uplift: DoWhy-based causal effect estimation for marketing uplift / churn interventions with refutation tests and counterfactual analysis.
- analytical/tabular/causal/econml_heterogeneous_effects: EconML heterogeneous treatment effect modeling for personalized policy decisions (Double ML, DR Learner) with confidence intervals.
- analytical/tabular/causal/causalml_campaign_optimization: CausalML uplift modeling for multi-treatment marketing campaigns and personalized engagement strategies.

## Time Series

- analytical/time_series/merlion_anomaly_dashboard: Salesforce Merlion anomaly detection + dashboard container.
- analytical/time_series/moirai_zero_shot: Zero-shot forecasting with Salesforce Moirai foundation model.
- analytical/time_series/uni2ts_finetune: Fine-tuning Salesforce Uni2TS on benchmark datasets.
- analytical/time_series/chronos_zero_shot: Amazon Chronos-Bolt/TinyChronos zero-shot multiseries forecasting with CLI notebook workflow.
- analytical/time_series/darts_tft_retail: Darts Temporal Fusion Transformer training for retail demand forecasting.
- analytical/time_series/neuralforecast_multiseries: Nixtla NeuralForecast (NHITS/TimesNet) multiseries training with GPU acceleration and evaluation suite.
- analytical/time_series/automl/autogluon_timeseries: Automated forecasting stacks with AutoGluon-TimeSeries or AutoTS baseline for quick leaderboard-style comparisons.
- analytical/time_series/gluonts_deepar: Probabilistic forecasting with GluonTS DeepAR, including SageMaker-compatible Dockerfile.
- analytical/time_series/prophet_business_forecasting: Classical Prophet-based business forecast baseline with reusable CLI.
- analytical/time_series/data_harmonization/pipeline: End-to-end preprocessing pipeline aligning multi-frequency, noisy time series before model training (resampling, denoising, missing value handling).
- analytical/time_series/hierarchical_reconciliation: Hierarchical forecasting with reconciliation (TopDown, MinT) using Nixtla or htsprophet on multi-level retail datasets.
- analytical/time_series/streaming_anomaly/merlion_online: Online/streaming anomaly detection leveraging Merlion’s online detectors or River-based AD models for live sensor feeds.
- analytical/time_series/imputation/denoising_autoencoder: Dedicated imputation and denoising module for irregular, noisy sensor time series prior to modeling.
- analytical/time_series/causal_impact/google: Causal impact / uplift analysis on time series interventions using Google CausalImpact or Nixtla’s causal forecasting toolkit.
- analytical/time_series/change_point_detection/kats: Change-point and trend detection using Salesforce Kats with visual diagnostics.
- analytical/time_series/classification/sktime_shapelets: Time series classification example using sktime shapelet-based models on benchmark datasets.
