#!/usr/bin/env python3
"""Simplify raw boundary GeoJSON into the small files used by the map.
Needs:  pip install shapely
Usage:  python3 tools/prepare_boundaries.py raw_country.geojson raw_states.geojson
  raw_country.geojson  one MultiPolygon feature for the whole country
  raw_states.geojson   a FeatureCollection with a NAME_1 property per state
Writes assets/geo/country.json and assets/geo/states.json, then run tools/build_geo.py.
"""
import json, sys, pathlib
from shapely.geometry import Polygon
root = pathlib.Path(__file__).resolve().parent.parent

def ring(g, tol, min_area):
    P = Polygon(g)
    if not P.is_valid:
        P = P.buffer(0)
    if P.area < min_area:
        return []
    S = P.simplify(tol, preserve_topology=False)
    out = []
    for q in (S.geoms if hasattr(S, "geoms") else [S]):
        if not q.is_empty and q.area >= min_area:
            out.append([[round(x, 2), round(y, 2)] for x, y in q.exterior.coords])
    return out

country = json.load(open(sys.argv[1]))
geom = country["features"][0]["geometry"] if country.get("features") else country["geometry"]
c = [r for poly in geom["coordinates"] for r in ring(poly[0], 0.025, 0.004)]
json.dump(c, open(root / "assets/geo/country.json", "w"), separators=(",", ":"))

s = []
for f in json.load(open(sys.argv[2]))["features"]:
    g = f["geometry"]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    s.append({"n": f["properties"]["NAME_1"], "r": [r for poly in polys for r in ring(poly[0], 0.04, 0.02)]})
json.dump(s, open(root / "assets/geo/states.json", "w"), separators=(",", ":"))
print("country parts:", len(c), "| states:", len(s))
