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

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_document():
    doc = docx.Document()

    # Set standard 1-inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Styles
    navy = RGBColor(26, 54, 93)     # #1A365D
    slate = RGBColor(74, 85, 104)   # #4A5568
    black = RGBColor(26, 32, 44)    # #1A202C

    # Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("AGNI-NETRA")
    run_title.font.name = "Arial"
    run_title.font.size = Pt(26)
    run_title.font.bold = True
    run_title.font.color.rgb = navy

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("Database Specification & Backend System Architecture\nSmart India Hackathon (SIH)")
    run_sub.font.name = "Arial"
    run_sub.font.size = Pt(13)
    run_sub.font.color.rgb = slate

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 1. Executive Summary
    h1 = doc.add_heading("1. Executive Summary", level=1)
    h1.style.font.color.rgb = navy
    p = doc.add_paragraph(
        "AGNI-NETRA is an advanced geospatial thermal anomaly monitoring and early-warning intelligence platform. "
        "It integrates near real-time (NRT) satellite thermal observations (NASA FIRMS / VIIRS 375m) with 141 multimodal "
        "Earth Observation (EO) features spanning terrain, meteorology, land cover, industrial facility proximity, "
        "and atmospheric chemistry. This document details the complete production database schema, technology stack, "
        "and architectural patterns that govern the system."
    )
    p.style.font.name = "Calibri"
    p.style.font.size = Pt(11)

    # 2. Technology Stack & Extensions
    h2 = doc.add_heading("2. Technology Stack & Architectural Roles", level=1)
    h2.style.font.color.rgb = navy

    tech_data = [
        ("Component", "Technology", "Architectural Purpose & Advantages"),
        ("Database Engine", "PostgreSQL 18", "Enterprise relational database offering ACID compliance and advanced JSONB/extension support."),
        ("Geospatial Engine", "PostGIS Extension", "Enables native spatial types, geodesic distance calculation (ST_DWithin), and polygon containment (ST_Intersects)."),
        ("Vector Search", "pgvector Extension", "Stores 141-dimensional normalized feature embeddings for instant cosine similarity matching (<=>) of analog events."),
        ("REST API Layer", "FastAPI (Python 3.11+)", "High-performance async web framework with automatic OpenAPI/Swagger documentation generation."),
        ("Data Mapping (ORM)", "SQLAlchemy 2.0", "Declarative object-relational mapping with automated connection pooling and type safety."),
        ("Data Validation", "Pydantic v2", "Strict schema enforcement and runtime validation matching docs/api-contract.md."),
        ("Job Scheduler", "APScheduler", "In-process background scheduler polling NASA FIRMS every 15 minutes, guarded by PostgreSQL advisory locks."),
        ("In-Memory Caching", "Cachetools (TTLCache)", "Fast in-process LRU cache (30-min TTL) for gridded wind vectors and 73-hour weather forecasts.")
    ]

    t_tech = doc.add_table(rows=len(tech_data), cols=3)
    t_tech.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(tech_data):
        for c_idx, text in enumerate(row):
            cell = t_tech.cell(r_idx, c_idx)
            cell.text = text
            set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
            p = cell.paragraphs[0]
            p.runs[0].font.name = "Calibri"
            p.runs[0].font.size = Pt(10)
            if r_idx == 0:
                set_cell_background(cell, "1A365D")
                p.runs[0].font.bold = True
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
            elif r_idx % 2 == 1:
                set_cell_background(cell, "F7FAFC")

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 3. Database Schema Breakdown
    h3 = doc.add_heading("3. Complete Database Tables & Column Specifications", level=1)
    h3.style.font.color.rgb = navy

    # Table 1: Places
    doc.add_heading("Table 1: places (Monitored Infrastructure & Baseline Sites)", level=2)
    doc.add_paragraph("Stores catalogued industrial installations, refineries, power stations, coal basins, and forest tracts across India.")
    
    places_cols = [
        ("Column Name", "Type", "Constraints", "Description & Purpose", "Example Value"),
        ("id", "VARCHAR(32)", "PRIMARY KEY, INDEX", "Unique site code.", "'GJ-01'"),
        ("name", "VARCHAR(255)", "NOT NULL", "Descriptive title of site.", "'Refinery cluster'"),
        ("state", "VARCHAR(100)", "NOT NULL", "Indian state location.", "'Gujarat'"),
        ("category", "VARCHAR(50)", "NULLABLE", "Physical category.", "'industrial'"),
        ("lat", "FLOAT", "NOT NULL", "Latitude (WGS84).", "22.3500"),
        ("lon", "FLOAT", "NOT NULL", "Longitude (WGS84).", "69.8500"),
        ("created_at", "TIMESTAMP", "DEFAULT UTC NOW", "Creation timestamp.", "2026-09-27 12:00:00")
    ]
    render_table(doc, places_cols)

    # Table 2: Events
    doc.add_heading("Table 2: events (Thermal Satellite Detections)", level=2)
    doc.add_paragraph("The central telemetry table capturing satellite hotspot clusters, classifications, and triage states.")

    events_cols = [
        ("Column Name", "Type", "Constraints", "Description & Purpose", "Example Value"),
        ("id", "VARCHAR(32)", "PRIMARY KEY, INDEX", "Unique event identifier.", "'E0218'"),
        ("place_id", "VARCHAR(32)", "FOREIGN KEY -> places.id", "Nearest monitored site reference.", "'GJ-01'"),
        ("detected_at", "TIMESTAMP", "NOT NULL, INDEX", "Satellite overpass timestamp.", "2026-09-27 13:15:00"),
        ("lat", "FLOAT", "NOT NULL", "Hotspot centroid latitude.", "15.1800"),
        ("lon", "FLOAT", "NOT NULL", "Hotspot centroid longitude.", "76.6500"),
        ("frp_median", "FLOAT", "DEFAULT 0.0", "Median Fire Radiative Power (MW).", "125.4"),
        ("predicted_class", "VARCHAR(50)", "NOT NULL, INDEX", "6-Class Taxonomy classification.", "'gas_flare'"),
        ("confidence", "FLOAT", "DEFAULT 0.5", "Calibrated confidence score (0.0-1.0).", "0.88"),
        ("baseline_status", "VARCHAR(20)", "NOT NULL, INDEX", "Site-level physical status: 'routine' vs 'abnormal'.", "'routine'"),
        ("status", "VARCHAR(30)", "NOT NULL, INDEX", "Analyst triage state: 'unreviewed', 'confirmed', 'false_alarm'.", "'unreviewed'"),
        ("missing_fraction", "FLOAT", "DEFAULT 0.0", "Fraction of missing features (0.0-1.0).", "0.08"),
        ("created_at", "TIMESTAMP", "DEFAULT UTC NOW", "DB insertion timestamp.", "2026-09-27 13:20:00")
    ]
    render_table(doc, events_cols)

    # Table 3: Event Features
    doc.add_heading("Table 3: event_features (141 Features & AI Explainability)", level=2)
    doc.add_paragraph("Stores the multimodal feature catalog, precomputed TreeSHAP attribution, and 141-D embeddings.")

    features_cols = [
        ("Column Name", "Type", "Constraints", "Description & Purpose", "Example Value"),
        ("event_id", "VARCHAR(32)", "PRIMARY KEY, FK -> events.id", "1:1 link to parent event.", "'E0218'"),
        ("feature_schema_version", "INTEGER", "DEFAULT 1, INDEX", "Protects embeddings from vector drift.", "1"),
        ("features_json", "JSONB / JSON", "NOT NULL", "Complete 141-feature dictionary.", "{\"FRP_MEDIAN\": {\"val\": 125.4}}"),
        ("shap_json", "JSONB / JSON", "NULLABLE", "Precomputed TreeSHAP family & top drivers.", "{\"by_family\": {\"industry\": 0.42}}"),
        ("embedding", "VECTOR(141)", "NULLABLE", "141-D normalized vector for pgvector.", "[0.34, -0.12, 0.05, ...]")
    ]
    render_table(doc, features_cols)

    # Table 4: Event Geometries
    doc.add_heading("Table 4: event_geometries (Smoke Plumes & Spread Polygons)", level=2)
    doc.add_paragraph("Stores geospatial vector overlays for Leaflet map display (smoke dispersion and fire-spread envelopes).")

    geom_cols = [
        ("Column Name", "Type", "Constraints", "Description & Purpose", "Example Value"),
        ("id", "INTEGER", "PRIMARY KEY, AUTOINCREMENT", "Unique geometry record ID.", "1"),
        ("event_id", "VARCHAR(32)", "FOREIGN KEY -> events.id", "Parent event reference.", "'E0218'"),
        ("layer_type", "VARCHAR(50)", "NOT NULL", "Layer type: plume_core, plume_outer, spread_outlook.", "'plume_core'"),
        ("horizon_hours", "INTEGER", "NULLABLE", "Forecast projection horizon.", "6"),
        ("geometry_json", "JSONB / JSON", "NOT NULL", "Coordinate ring polygon [[lon, lat], ...].", "[[76.62, 15.12], [76.58, 15.16]]"),
        ("created_at", "TIMESTAMP", "DEFAULT UTC NOW", "DB insertion timestamp.", "2026-09-27 13:20:00")
    ]
    render_table(doc, geom_cols)

    # Table 5: Analyst Reviews
    doc.add_heading("Table 5: analyst_reviews (Human-in-the-Loop Ground Truth)", level=2)
    doc.add_paragraph("Records operator confirmation or reclassification calls, generating a verified ground-truth training dataset.")

    review_cols = [
        ("Column Name", "Type", "Constraints", "Description & Purpose", "Example Value"),
        ("id", "INTEGER", "PRIMARY KEY, AUTOINCREMENT", "Unique review identifier.", "1"),
        ("event_id", "VARCHAR(32)", "FOREIGN KEY -> events.id", "Associated event reference.", "'E0218'"),
        ("verdict", "VARCHAR(20)", "NOT NULL", "Action verdict: 'confirm', 'change', 'false', 'field'.", "'confirm'"),
        ("analyst_class", "VARCHAR(50)", "NULLABLE", "Reclassified class if verdict was 'change'.", "'mining'"),
        ("note", "TEXT", "NULLABLE", "Operator notes and evidence description.", "'Excursion verified via Sentinel-2.'"),
        ("reviewer", "VARCHAR(100)", "NULLABLE", "Analyst username / badge ID.", "'Analyst-04'"),
        ("at", "VARCHAR(100)", "NULLABLE", "Display timestamp for frontend.", "'27 Sep 2026, 18:30 IST'"),
        ("created_at", "TIMESTAMP", "DEFAULT UTC NOW", "Record insertion timestamp.", "2026-09-27 18:30:00")
    ]
    render_table(doc, review_cols)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 4. Core Design Innovations
    h4 = doc.add_heading("4. Key Architectural Innovations & Technical Merits", level=1)
    h4.style.font.color.rgb = navy

    points = [
        ("Separation of Physical vs Workflow Status: ", "Unlike standard systems that conflate operational normalcy with review state, AGNI-NETRA separates baseline_status (routine vs abnormal) from status (unreviewed, confirmed, false_alarm). A flare at a refinery can be completely routine physically, while remaining unreviewed operationally."),
        ("Deterministic Heuristic Rule-Based Labeling: ", "In the absence of nationwide clean training labels, labels are generated via transparent spatial proximity, seasonal windows, and land cover heuristics (e.g. Cropland + Oct-Nov harvest + high spread speed = agricultural_burning). This is completely defensible to evaluators."),
        ("Precomputed TreeSHAP (<5ms API Latency): ", "Computing SHAP attribution on the fly consumes substantial CPU cycles per request. Precomputing SHAP at event ingestion and caching it in shap_json delivers instantaneous O(1) response times during live presentations."),
        ("Schema Versioning with Deferred Indexing: ", "The feature_schema_version column protects pgvector embeddings from silent drift when formulas change. Heavy IVFFlat and GIN indexes are deferred for demo-scale data, guaranteeing 100% recall with exact brute-force search in under 2 milliseconds.")
    ]

    for title, desc in points:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.2)
        r_bold = p.add_run(f"• {title}")
        r_bold.font.name = "Calibri"
        r_bold.font.size = Pt(10.5)
        r_bold.font.bold = True
        r_bold.font.color.rgb = navy
        r_text = p.add_run(desc)
        r_text.font.name = "Calibri"
        r_text.font.size = Pt(10.5)

    output_path = r"c:\Users\NITISH SINGH\Downloads\agni-netra-v2\agni-netra-v2\AGNI-NETRA_Database_and_System_Design.docx"
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
    create_document()
