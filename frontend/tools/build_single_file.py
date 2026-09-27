#!/usr/bin/env python3
"""Build one self-contained HTML file from the project.
Usage (from the project root):
  python3 tools/build_single_file.py                      -> dist/agni-netra-single.html (charts load from a CDN)
  python3 tools/build_single_file.py --inline-echarts     -> also inlines the chart library, so it works offline
  python3 tools/build_single_file.py my-name.html         -> custom output path
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
inline = "--inline-echarts" in sys.argv
out = pathlib.Path(args[0]) if args else root / "dist" / "agni-netra-single.html"
read = lambda p: (root / p).read_text(encoding="utf-8")
import re
html = read("index.html")
html = html.replace('<link rel="stylesheet" href="vendor/leaflet/leaflet.css">', "<style>\n" + read("vendor/leaflet/leaflet.css") + "\n</style>")
for css in ["css/base.css", "css/analysis.css"]:
    html = html.replace(f'<link rel="stylesheet" href="{css}">', "<style>\n" + read(css) + "\n</style>")
html = html.replace('<script src="vendor/leaflet/leaflet.js"></script>', "<script>\n" + read("vendor/leaflet/leaflet.js") + "\n</script>")
html = html.replace('<script src="vendor/echarts.min.js"></script>', ("<script>\n" + read("vendor/echarts.min.js") + "\n</script>") if inline
                    else '<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>')
for m in re.findall(r'<script src="(js/[^"]+\.js)"></script>', html):
    html = html.replace(f'<script src="{m}"></script>', "<script>\n" + read(m) + "\n</script>")
# Leaflet's marker images are referenced by url(images/...) relative to leaflet.css; inline them as data URIs
import base64, pathlib
def data_uri(path):
    b = (root / path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()
for name in ["marker-icon.png", "marker-icon-2x.png", "marker-shadow.png"]:
    html = html.replace(f"images/{name}", data_uri(f"vendor/leaflet/images/{name}"))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html, encoding="utf-8")
print("wrote", out, len(html) // 1024, "KB")
