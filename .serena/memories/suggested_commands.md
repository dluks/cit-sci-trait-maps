## Environment setup
```bash
# Using Poetry
poetry install
poetry run python -V

# Using conda (alternative)
conda env create -f environment.yml && conda activate cit-sci-traits

# DVC (recommended isolated install)
pipx install dvc
```

## Dependency management
- Poetry (preferred for this repo):
```bash
poetry add <package>
```
- If using uv (your preference):
```bash
uv add <package>
```

## Containers (Singularity/Apptainer)
```bash
# Build via DVC stage (recommended)
dvc repro build_container
# Or direct build
apptainer build cit-sci-traits.sif cit-sci-traits.def
```

## Data & pipeline
```bash
dvc pull                # fetch data artifacts
dvc repro               # run full pipeline
dvc repro <stage-name>  # run one stage
```
Common stages: `harmonize_eo_data`, `build_predict`, `build_y`, `train_models`, `aoa`, `predict`, `cov`, `build_final_product`.

## Running modules directly
```bash
poetry run python src/models/train_models.py
poetry run python src/models/predict_traits.py -r -v
```

## Testing, linting, typing, formatting
```bash
poetry run pytest -q
poetry run pylint src tests
poetry run pyright
poetry run black .
```

## Notebooks
```bash
poetry run jupyter lab
```

## Useful Linux commands
```bash
ls -la; pwd; du -sh *; nproc; nvidia-smi
rg -n "pattern"            # ripgrep if available
find . -name "*.py" | wc -l
```

## HPC/Container execution examples
```bash
apptainer exec --nv cit-sci-traits.sif poetry run python src/models/train_models.py
```