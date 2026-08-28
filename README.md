# Soccer Match Prediction using Multimodal Neural Networks

## Getting Started

To scrape the project data, run:

```bash
./.venv/bin/python run_data_pipeline.py module1
./.venv/bin/python run_data_pipeline.py module2
./.venv/bin/python run_data_pipeline.py module3
```

Choose the module you need:

- `module1` for player-statistics and lineup data (needs one-time internet connection)
- `module2` for match-statistics data (needs permanent internet connection)
- `module3` for lineup embeddings

The scripts write their outputs into the existing data folders inside the repo.


