# LightGBM Ranking (LambdaMART) Demo

Train a LambdaMART ranker with LightGBM on the publicly available LETOR-style learning-to-rank toy dataset bundled with the LightGBM project. The pipeline downloads the dataset into the shared CVLization cache, trains a group-aware model, evaluates NDCG/MAP, and surfaces ranked predictions for sample queries.

## Dataset

- **Source**: LightGBM examples repository — [examples/lambdarank](https://github.com/microsoft/LightGBM/tree/master/examples/lambdarank)
- **Files used**: `rank.train`, `rank.valid`, `rank.test` plus their corresponding `.query` files
- **Caching**: Files are stored in `~/.cache/cvlization/data/lightgbm_lambdarank_demo`

## Quickstart

```bash
# From repo root
cvl run ranking-lightgbm build
cvl run ranking-lightgbm train
cvl run ranking-lightgbm predict -- --input artifacts/sample_candidates.csv --output outputs/sample_rankings.csv
```

Environment variables such as `NUM_BOOST_ROUND` and `BOOSTING_TYPE` can be set before `train` to tweak training speed or model type.

## Artifacts

- `artifacts/ranking_lightgbm.txt`: Trained LightGBM booster
- `artifacts/metrics.json`: NDCG@5, NDCG@10, MAP@10, best iteration
- `artifacts/feature_names.json`: Feature ordering required for inference
- `artifacts/sample_candidates.csv`: Example candidate rows (query-id + features)
- `artifacts/sample_rankings.csv`: Ranked results on the sample queries with relevance labels

## References

- [LightGBM LambdaRank demo](https://github.com/microsoft/LightGBM/tree/master/examples/lambdarank)
- [LightGBM ranker parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective-parameters)
