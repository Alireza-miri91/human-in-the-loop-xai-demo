# Human-in-the-Loop XAI Study Demo

This repository is a sanitized portfolio version of a Flask study app for
human-in-the-loop explainable AI.

The app demonstrates how users can inspect an interpretable neural additive
model and, in the treatment flow, apply monotonic domain constraints before
retraining the model.

## What This Shows

- A Flask router that assigns visitors to a control or treatment flow.
- A control interface with an interpretable baseline model.
- A treatment interface where users can request monotonic feature constraints.
- An IGANN-style model implementation with CVXPY-based constrained training.
- Bike-sharing demand prediction as the demo task.

## Privacy And Sanitization

This public version does not include real study data.

Removed or excluded from version control:

- participant response JSON files
- survey result CSV files
- assignment and engagement logs
- cookies and session dumps
- IP addresses and user-agent logging
- private analysis notebooks and generated reports
- private machine paths and backup artifacts

Any local demo submissions are written to `runtime_data/`, which is ignored by
git.

The included `data/day.csv` file is the public Bike Sharing Dataset commonly
used for ML demos and teaching examples. It is separate from the human-subject
study responses, which are intentionally excluded.

## Repository Structure

```text
.
├── data/
│   └── day.csv
├── runtime_data/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── control_app.py
│   ├── igann.py
│   ├── router_app.py
│   └── treatment_app.py
├── tests/
│   └── smoke_test.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

Start the router app:

```bash
python -m src.router_app
```

Then open:

```text
http://127.0.0.1:8050/
```

Useful public demo endpoints:

- `/` assigns the browser to a demo group.
- `/control/` opens the control flow.
- `/treatment/` opens the treatment flow.
- `/metrics` shows local assignment counts.
- `/health` returns a simple health check.

Admin/debug/reset routes are disabled by default in this public demo.

## Configuration

Optional environment variables:

- `HOST`: host for local serving, default `127.0.0.1`.
- `PORT`: port for local serving, default `8050`.
- `ROUTER_SECRET_KEY`: router Flask secret key.
- `HIL_XAI_APP_SECRET_KEY`: control/treatment Flask secret key.
- `HIL_XAI_RUNTIME_DATA_DIR`: directory for ignored local runtime artifacts.
- `HIL_XAI_DATA_PATH`: path to the bike-sharing CSV.
- `HIL_XAI_ENABLE_ADMIN_ROUTES`: set to `true` only for local debugging.

## Notes For Recruiters

This project is intended to show practical ML engineering around interpretable
models, constrained optimization, Flask app design, and responsible handling of
study data. The public repo keeps the runnable application and removes private
participant artifacts.
