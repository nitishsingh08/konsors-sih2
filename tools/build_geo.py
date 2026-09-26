#!/usr/bin/env python3
"""Regenerate js/data/geo.js from assets/geo/country.json and assets/geo/states.json.
Usage (from the project root):  python3 tools/build_geo.py
"""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
country = (root / "assets/geo/country.json").read_text(encoding="utf-8")
states = (root / "assets/geo/states.json").read_text(encoding="utf-8")
out = ("'use strict';\n"
       "/* Simplified India outline and state borders, drawn on the map for display only.\n"
       "   Not an authoritative boundary. See README.md, section 'Boundaries'. */\n"
       f"const COUNTRY = {country};\nconst STATES = {states};\n")
(root / "js/data/geo.js").write_text(out, encoding="utf-8")
print("wrote js/data/geo.js", len(out) // 1024, "KB")
