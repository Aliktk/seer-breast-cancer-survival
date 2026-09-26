# Deployment

## 1. Run locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app opens at http://localhost:8501. It only needs `requirements.txt`; the notebooks' extra libraries are not required.

## 2. Streamlit Community Cloud (free)

1. Push the repository to GitHub (public, or private with Streamlit access granted).
2. Sign in at https://share.streamlit.io with your GitHub account.
3. Click **Create app → Deploy a public app from GitHub**.
4. Fill in:
   - **Repository:** `Aliktk/seer-breast-cancer-survival`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
5. Open **Advanced settings** and choose **Python 3.11** (tested). The pinned versions also have wheels for 3.10–3.13.
6. Click **Deploy**. The first build takes a few minutes while `scikit-survival` installs.

Model files in `models/` are small (under 10 kB each for the selected models), so they are committed to the repository and loaded at start-up. No external storage is needed.

## 3. Updating the model

```bash
pip install -r requirements-dev.txt
python scripts/train.py          # rewrites models/*.joblib and models/metadata.json
pytest                           # inference tests must still pass
git add models/ reports/tables/ && git commit -m "Retrain models" && git push
```

Community Cloud redeploys automatically on push.

## 4. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: seer_survival` | app started from a different working directory | the app adds `src/` to `sys.path` itself; make sure the main file path is `app/streamlit_app.py` |
| Error unpickling `*.joblib` | scikit-learn / scikit-survival version differs from training | install the exact versions in `requirements.txt`, or retrain |
| Build fails on numpy / scikit-survival | unsupported Python version | choose Python 3.11 in Advanced settings |
