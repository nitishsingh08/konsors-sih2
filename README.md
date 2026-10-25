# 🛰️ AGNI-NETRA

> **AI-driven fire-event classification & monitoring platform**  
> Detects industrial vs. natural fire events from satellite & sensor data, combining geospatial intelligence with machine learning.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Running the Project](#running-the-project)
- [Tests](#tests)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

AGNI-NETRA is a multi-stage pipeline that:

1. **Ingests** fire-event detections from satellite (MODIS/VIIRS) and ground sources.
2. **Extracts** 141+ features per event (thermal, temporal, terrain, landcover, industrial, weather, SAR, atmospheric).
3. **Trains** two LightGBM models:
   - **Model A** — operational model using all features (includes facility-proximity leak features).
   - **Model B** — scientific model using only non-leak features (no industrial proximity).
4. **Evaluates** with rigorous out-of-time × out-of-space splits and threshold calibration.
5. **Serves** predictions via a FastAPI backend with PostgreSQL + PostGIS.

---

## 📁 Project Structure

```
konsors-sih2/
├── backend/                 # FastAPI backend service
│   ├── requirements.txt     # Backend-only dependencies
│   ├── app/                 # Application code
│   └── tests/               # Backend tests
├── Colab_Files/             # ML pipeline notebooks (P0–P8)
├── Colab_helpers/           # Shared Python modules
│   ├── feature_ledger.py
│   ├── training_design.py
│   ├── feature_ledger.py
│   └── ...
├── Features_mapped/         # Large dataset (hosted on Google Drive)
│   └── ablation_table.csv
├── Heldout_csv/             # Held-out predictions
├── models/                  # Saved model pickles
├── qc/                      # Quality control outputs & figures
├── test/                    # Pipeline tests
├── requirements.txt         # Full project dependencies
├── setup.sh                 # macOS / Linux setup
├── setup.ps1                # Windows setup
└── README.md
```

---

## ⚙️ Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| **Python** | ≥ 3.10 | Core runtime |
| **7-Zip** | ≥ 22.0 | Extracting dataset archive |
| **pip** | latest | Package management |
| **PostgreSQL** + **PostGIS** | ≥ 14 / ≥ 3.3 | Backend database (optional) |
| **Git** | ≥ 2.30 | Version control |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/konsors_sih2.git
cd konsors_sih2
```

### 2. Set up the environment

**macOS / Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

### 3. Install Python dependencies

```bash
# Full project (ML + backend)
pip install -r requirements.txt

# Backend only
pip install -r backend/requirements.txt
```

### 4. (Optional) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

---

## 📦 Dataset Setup

The `Features_mapped` dataset is **hosted on Google Drive** due to its large size and is **not stored directly** in this Git repository.

### Automatic Download

The `setup.sh` / `setup.ps1` scripts attempt to download the dataset automatically. If that fails, follow the manual steps below.

### Manual Download

1. **Download** `Features_mapped.7z` from the [Google Drive dataset link](https://drive.google.com/drive/folders/PLACEHOLDER).
2. **Extract** the archive into the repository root using 7-Zip:

   **macOS / Linux:**
   ```bash
   7z x Features_mapped.7z -o./
   ```

   **Windows (PowerShell):**
   ```powershell
   7z x Features_mapped.7z -o.
   ```

3. Verify the extracted directory structure:
   ```
   konsors_sih2/Features_mapped/ablation_table.csv
   ```

### ⚠️ Requirements

- **7-Zip** installed and available on your `PATH`.
- **Sufficient disk space** for both the archive and the extracted files (estimate ~2–5 GB).
- The dataset is **required only** for workflows that use the `Features_mapped` data (model training, ablation, SHAP analysis).

---

## ▶️ Running the Project

### Backend (FastAPI)

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Access the API docs at: `http://localhost:8000/docs`

### ML Pipeline (Colab / Local)

The ML pipeline is organized as sequential stages (P0 → P8):

```bash
# 1. Bootstrap & feature audit
python Colab_Files/colab_bootstrap.py

# 2. Train models (A + B)
python Colab_Files/colab_p7_train.py

# 3. SHAP analysis
python Colab_Files/colab_p7m_shap_analysis.py

# 4. Leakage probe
python Colab_Files/colab_p8b_leakage.py

# 5. Generate figures
python Colab_Files/colab_p8_figures.py
```

### Retraining with ablation

```bash
python Colab_helpers/ablation_table.py
```

### Evaluate held-out predictions

```bash
python Colab_helpers/scripts/evaluate_model.py --scores qc/heldout_pred_B.csv
```

---

## 🧪 Tests

Run the full test suite:

```bash
# All tests
python -m pytest test/ -v

# Backend tests
python -m pytest backend/tests/ -v

# Specific module
python -m pytest test/test_training_design.py -v
```

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m "Add your feature"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a Pull Request.

### Guidelines

- Follow the existing code style (no unnecessary comments).
- Add tests for new functionality.
- Run the full test suite before submitting.
- Do not commit secrets, `.env` files, or large datasets.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

<div align="center">

**Built with ❤️ for SIH 2026**

</div>
