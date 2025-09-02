```
root
├── data/               # raw/interim/processed/features (see params.yaml)
├── models/             # trained models per PFT/resolution
├── notebooks/          # organized by analysis stage
├── reference/          # trait mappings and metadata
├── results/            # analysis outputs and figures
├── src/
│   ├── analysis/
│   ├── data/
│   ├── features/
│   ├── io/
│   ├── models/
│   ├── utils/
│   └── visualization/
├── stages/             # shell wrappers for some DVC stages
├── container/ | .apptainer/ | cit-sci-traits.def
├── dvc.yaml, dvc.lock  # pipeline and lock
├── params.yaml         # parameters & resource knobs
├── pyproject.toml      # Poetry config and deps
└── tests/              # pytest tests
```