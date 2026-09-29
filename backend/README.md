# AGNI-NETRA: FastAPI Backend Service

Production-ready backend API supporting the **AGNI-NETRA v2** dashboard.

---

## 1. Quick Start (Local Development)

### Prerequisites
- Python 3.10+ installed

### Setup & Run
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the server (pinned to a single worker):
   ```bash
   uvicorn app.main:app --reload --workers 1 --port 8000
   ```

5. Open interactive API docs:
   - **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
   - **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 2. Connecting to PostgreSQL with PostGIS & pgvector

For production or cloud deployment (e.g. Supabase, Neon, AWS RDS, Local):
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Set your `DATABASE_URL` in `.env`:
   ```env
   DATABASE_URL="postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5432/SIH"
   ```
3. Enable PostGIS and vector extensions on your database:
   ```sql
   CREATE EXTENSION IF NOT EXISTS postgis;
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
4. Seed initial places and events:
   ```bash
   python -m app.seed
   ```

---

## 3. Endpoints Implemented

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/events` | GET | List events with 6-class taxonomy, `baseline_status`, and review `status`. |
| `/api/events/{id}` | GET | Fetch single event details. |
| `/api/events/{id}/features` | GET | 141 model features dictionary. |
| `/api/events/{id}/shap` | GET | Precomputed TreeSHAP attribution scores. |
| `/api/events/{id}/forecast` | GET | 73-hour weather & Canadian FWI forecast. |
| `/api/wind` | GET | Gridded $u$ and $v$ wind velocity vectors. |
| `/api/events/{id}/plume` | GET | 6-hour Gaussian smoke dispersion trajectory and polygon. |
| `/api/events/{id}/spread` | GET | 1h, 3h, 6h Huygens elliptical fire-front outlook. |
| `/api/events/{id}/exposure` | GET | Critical infrastructure & population exposure counts. |
| `/api/events/{id}/similar` | GET | Vector similarity ranking of historical analog events. |
| `/api/events/{id}/reviews` | POST | Persist analyst ground truth confirmations/corrections. |
| `/api/reviews` | GET | Retrieve analyst review history. |
