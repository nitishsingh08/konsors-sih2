import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_walkthrough_document():
    doc = docx.Document()

    # 1-inch margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Color palette
    navy = RGBColor(26, 54, 93)     # #1A365D Primary
    slate = RGBColor(74, 85, 104)   # #4A5568 Secondary
    dark = RGBColor(45, 55, 72)     # #2D3748 Body text

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("SPARC")
    run_title.font.name = "Arial"
    run_title.font.size = Pt(26)
    run_title.font.bold = True
    run_title.font.color.rgb = navy

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("Complete Backend Architecture & System Design Walkthrough\nSmart India Hackathon (SIH)")
    run_sub.font.name = "Arial"
    run_sub.font.size = Pt(13)
    run_sub.font.color.rgb = slate

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Section 1: Core Design Philosophy
    h1 = doc.add_heading("1. Core Design Philosophy & Operational Context", level=1)
    h1.style.font.color.rgb = navy
    p1 = doc.add_paragraph(
        "Satellite thermal monitoring across India faces a major operational dilemma: at 375-meter resolution, "
        "a raw thermal anomaly looks virtually identical whether it is a wildfire in an Uttarakhand forest, "
        "crop stubble burning in Punjab, a continuous petrochemical flare in Jamnagar, or an underground coal fire in Jharia. "
        "Standard backends fail because they treat satellite hotspots as simple isolated points. SPARC's backend was "
        "specifically architected to resolve this through four fundamental design tenets:"
    )
    p1.style.font.name = "Calibri"
    p1.style.font.size = Pt(11)

    tenets = [
        ("Multimodal Feature Fusion: ", "Combines raw thermal pixels with 141 scientific features spanning terrain, meteorology, land cover, industrial facility proximity, and radar backscatter."),
        ("Deterministic 6-Class Taxonomy: ", "Classifies thermal events transparently into wildfire, agricultural_burning, gas_flare, industrial, mining, and unknown using defensible heuristic rules."),
        ("Instant Explainable AI (XAI): ", "Precomputes TreeSHAP feature attributions at event ingestion time, delivering sub-5ms explainability lookups without dashboard latency."),
        ("Sub-15ms Operational Speed: ", "Utilizes in-process LRU caching (TTLCache) and optimized database querying, eliminating external message broker dependencies during live presentations.")
    ]
    for title, desc in tenets:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.2)
        rb = p.add_run(f"• {title}")
        rb.font.bold = True
        rb.font.color.rgb = navy
        p.add_run(desc)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Section 2: Directory Structure
    h2 = doc.add_heading("2. Backend Directory Structure & Organization", level=1)
    h2.style.font.color.rgb = navy

    p2 = doc.add_paragraph(
        "The backend is modularized under the backend/app/ directory, cleanly separating database configuration, "
        "data models, API schemas, heuristic services, and REST routing:"
    )
    p2.style.font.name = "Calibri"

    tree_data = [
        ("Directory / File", "Architectural Role", "Key Responsibilities"),
        ("backend/app/core/config.py", "Configuration", "Loads .env variables, defines CORS origins, schema versioning, and polling intervals."),
        ("backend/app/core/database.py", "Database Engine", "SQLAlchemy connection manager with auto-dialect normalization (postgresql+psycopg2) and SQLite fallback."),
        ("backend/app/models/models.py", "ORM Schema", "5 core relational models: Place, Event, EventFeatures, EventGeometry, AnalystReview."),
        ("backend/app/schemas/schemas.py", "Validation", "Pydantic models enforcing strict contract compliance with docs/api-contract.md."),
        ("backend/app/services/heuristic_rules.py", "Rules Engine", "Deterministic spatial & seasonal rules mapping hotspots to the 6-class taxonomy."),
        ("backend/app/services/ml_engine.py", "AI & Embeddings", "Precomputes TreeSHAP feature attributions and generates normalized 141-D vectors for pgvector."),
        ("backend/app/services/weather_cache.py", "In-Process Cache", "TTLCache storing 45x45 gridded wind vectors and 73-hour weather forecasts in RAM (no Redis needed)."),
        ("backend/app/services/scheduler.py", "Task Scheduler", "APScheduler background runner polling NASA FIRMS NRT feeds every 15 mins with DB advisory locking."),
        ("backend/app/api/routes.py", "REST Routing", "Implements all 10 endpoints matching the frontend dashboard contract."),
        ("backend/app/main.py", "Application Entry", "FastAPI app instance, lifespan startup/shutdown management, and CORS middleware."),
        ("backend/app/seed.py", "Database Seeder", "Connects to PostgreSQL SIH database and populates persistent places and active thermal events.")
    ]
    render_table(doc, tree_data)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Section 3: Detailed Module Breakdown
    h3 = doc.add_heading("3. Detailed Module-by-Module Technical Breakdown", level=1)
    h3.style.font.color.rgb = navy

    modules = [
        ("Module 1: Dual-Dialect Database Manager (database.py)",
         "Configured with SQLAlchemy 2.0. If a standard 'postgresql://' URL is provided, the engine automatically normalizes "
         "it to 'postgresql+psycopg2://' to prevent dialect mismatches. Configures connection pre-pinging (pool_pre_ping=True) "
         "to prevent stale connection errors during long-running server sessions."),

        ("Module 2: 5 Relational & Geospatial ORM Tables (models.py)",
         "Defines 5 core tables:\n"
         "1. places: Catalogues monitored industrial sites, refineries, power plants, and coal basins.\n"
         "2. events: Stores satellite detections, linking coordinates, FRP heat intensity, 6-class taxonomy, baseline_status, and analyst status.\n"
         "3. event_features: Stores 141 features in JSONB, precomputed SHAP attribution, and 141-D vector embeddings.\n"
         "4. event_geometries: Stores vector coordinates for 6-hour smoke plume cones and 1h/3h/6h fire spread ellipses.\n"
         "5. analyst_reviews: Stores human operator triage calls, creating a persistent ground-truth training set."),

        ("Module 3: Deterministic Heuristic Labeling Engine (heuristic_rules.py)",
         "Rather than relying on unverified synthetic labels, the system evaluates transparent rules based on spatial proximity, "
         "season, and land cover:\n"
         "• gas_flare: Inside refinery or <1km distance with stationarity >= 0.85 (baseline: 'routine' if recurrent, confidence: 0.84 - 0.92).\n"
         "• industrial: Inside GEM industrial boundary with stationarity >= 0.75 (confidence: 0.86).\n"
         "• agricultural_burning: Cropland fraction >= 40% during harvest window (Oct-Nov / Apr-May) with spread speed >= 0.1 km/h (confidence: 0.80 - 0.88).\n"
         "• wildfire: Forest fraction >= 35% or elevated Fire Weather Index (FWI >= 20) (confidence: 0.81 - 0.89).\n"
         "• mining: Distance to active mine <= 1.5km or bare ground >= 50% (confidence: 0.82)."),

        ("Module 4: Precomputed TreeSHAP & Vector Embeddings (ml_engine.py)",
         "Computing TreeSHAP over 141 features on every live GET request causes 20-50ms CPU latency. To ensure rock-solid demo performance, "
         "the ML engine calculates SHAP family contributions and top-5 drivers once upon event ingestion and persists them into "
         "event_features.shap_json. Serving /api/events/{id}/shap becomes an instantaneous O(1) lookup under 5 milliseconds. "
         "Also normalizes all 141 features into a standardized 141-D vector for pgvector cosine similarity searches."),

        ("Module 5: In-Process LRU Weather & Wind Cache (weather_cache.py)",
         "Uses cachetools.TTLCache (30-minute expiration) directly inside application memory. Caches 45x45 gridded u/v wind components "
         "and 73-hour Canadian Fire Weather Index forecasts (FFMC, DMC, DC, ISI, BUI, FWI). Eliminates the need to install and manage "
         "a separate Redis instance for hackathon deployments."),

        ("Module 6: Concurrency-Guarded Task Scheduler (scheduler.py)",
         "Runs an in-process APScheduler background job polling NASA FIRMS VIIRS 375m active fires every 15 minutes. "
         "Guarded by a PostgreSQL session advisory lock (pg_try_advisory_lock(849201)): even if multiple workers are deployed, "
         "only one process polls NASA FIRMS, preventing duplicate database writes and API quota exhaustion.")
    ]

    for title, text in modules:
        doc.add_heading(title, level=2)
        p = doc.add_paragraph(text)
        p.style.font.name = "Calibri"
        p.style.font.size = Pt(10.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Section 4: Complete REST API Endpoints
    h4 = doc.add_heading("4. Complete REST API Specifications", level=1)
    h4.style.font.color.rgb = navy
    p4 = doc.add_paragraph("All 10 endpoints implemented in backend/app/api/routes.py strictly conform to docs/api-contract.md:")
    p4.style.font.name = "Calibri"

    api_data = [
        ("Endpoint Route", "HTTP", "Source & Logic", "Latency Budget"),
        ("/api/events", "GET", "List satellite events with 6-class taxonomy, baseline_status, and status.", "< 15 ms"),
        ("/api/events/{id}", "GET", "Fetch single event telemetry and classification.", "< 10 ms"),
        ("/api/events/{id}/features", "GET", "Returns 141-feature dictionary with real values, percentiles, and missing flags.", "< 10 ms"),
        ("/api/events/{id}/shap", "GET", "Instant lookup from precomputed event_features.shap_json.", "< 5 ms"),
        ("/api/events/{id}/forecast", "GET", "73-hour hourly weather and Canadian FWI forecast timeline.", "< 5 ms"),
        ("/api/wind", "GET", "Gridded u and v wind velocity vectors for Leaflet particle animation.", "< 10 ms"),
        ("/api/events/{id}/plume", "GET", "6-hour Gaussian smoke dispersion trajectory and polygon.", "< 15 ms"),
        ("/api/events/{id}/spread", "GET", "1h, 3h, 6h Huygens elliptical fire spread outlook for wildfire/stubble.", "< 20 ms"),
        ("/api/events/{id}/exposure", "GET", "PostGIS spatial intersection counting schools, clinics, settlements in plume.", "< 25 ms"),
        ("/api/events/{id}/similar", "GET", "Exact pgvector cosine similarity (<=>) ranking past analog fires.", "< 10 ms"),
        ("/api/events/{id}/reviews", "POST", "Persists analyst ground-truth confirm/reject/reclassify verdicts.", "< 10 ms"),
        ("/api/reviews", "GET", "Retrieves complete analyst triage and verification history.", "< 10 ms")
    ]
    render_table(doc, api_data)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Section 5: End-to-End Operational Lifecycle
    h5 = doc.add_heading("5. End-to-End Operational Lifecycle", level=1)
    h5.style.font.color.rgb = navy

    lifecycle = [
        ("1. Ingestion: ", "NASA FIRMS VIIRS 375m sensor detects a thermal anomaly over Karnataka."),
        ("2. Spatial Feature Evaluation: ", "System calculates proximity to nearest GEM industrial site (3.8 km), forest cover (45%), cropland (10%), and current FWI (12.0)."),
        ("3. Classification & Baseline: ", "Heuristic rules label the event as 'wildfire' with 88% calibrated confidence and 'abnormal' baseline status."),
        ("4. Explainability Generation: ", "TreeSHAP attribution weights and 141-D vector embeddings are precomputed in memory."),
        ("5. Database Commit: ", "All telemetry, features, and geometries are persisted into PostgreSQL tables in a single transaction."),
        ("6. Dashboard Delivery: ", "Leaflet frontend fetches /api/events in under 15ms, plotting the active fire marker, smoke plume envelope, and SHAP cards.")
    ]
    for step, desc in lifecycle:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.2)
        rb = p.add_run(step)
        rb.font.bold = True
        rb.font.color.rgb = navy
        p.add_run(desc)

    output_path = r"c:\Users\NITISH SINGH\Downloads\agni-netra-v2\agni-netra-v2\SPARC_Backend_Complete_Design_Walkthrough.docx"
    doc.save(output_path)
    print(f"Document successfully created at: {output_path}")

def render_table(doc, data):
    table = doc.add_table(rows=len(data), cols=len(data[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(data):
        for c_idx, text in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = text
            set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            p = cell.paragraphs[0]
            p.runs[0].font.name = "Calibri"
            p.runs[0].font.size = Pt(9.5)
            if r_idx == 0:
                set_cell_background(cell, "1A365D")
                p.runs[0].font.bold = True
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
            elif r_idx % 2 == 1:
                set_cell_background(cell, "F7FAFC")

if __name__ == "__main__":
    create_walkthrough_document()
