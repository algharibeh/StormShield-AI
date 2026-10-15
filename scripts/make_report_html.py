"""Convert the Markdown report into a standalone HTML page next to it.

The HTML uses the MP4 animation when present (crisper and smaller than the
GIF), with the GIF as fallback, and plain relative paths so it opens offline
from a git clone. A `<name>_last.png` still in the Markdown (the printable
copy of an animation) is swapped for `<name>.mp4` in the same way.

Usage: python scripts/make_report_html.py --md "visualizing data/REPORT.md"
"""
import argparse
import os
import re

import markdown

STYLE = """body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:32px auto;padding:0 20px;color:#222;line-height:1.5}
table{border-collapse:collapse;margin:12px 0;font-size:14px}th,td{border:1px solid #ddd;padding:5px 9px}th{background:#f4f4f4}
img,video{max-width:100%}code{background:#f4f4f4;padding:1px 4px}pre{background:#f4f4f4;padding:10px;overflow:auto}
h1{font-size:28px}h2{margin-top:34px;border-bottom:1px solid #eee}em{color:#444}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="visualizing data/REPORT.md")
    args = ap.parse_args()
    folder = os.path.dirname(args.md)
    src = open(args.md, encoding="utf-8").read()
    body = markdown.markdown(src, extensions=["tables", "fenced_code"])
    # swap each GIF image, or `_last.png` still, for a video element when an MP4 twin exists
    def swap(m):
        src, ext = m.group(1), m.group(2)
        base = src[:-5] if src.endswith("_last") else src
        mp4 = base + ".mp4"
        if os.path.exists(os.path.join(folder, mp4)):
            gif = base + ".gif"
            fallback = gif if os.path.exists(os.path.join(folder, gif)) else src + "." + ext
            return (f'<video autoplay loop muted playsinline controls poster="" '
                    f'style="width:100%"><source src="{mp4}" type="video/mp4">'
                    f'<img src="{fallback}" alt="animation"></video>')
        return m.group(0)
    body = re.sub(r'<img alt="[^"]*" src="([^"]+)\.(gif|png)"\s*/?>', swap, body)
    title = re.search(r"<h1>(.*?)</h1>", body).group(1)
    html = f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>{STYLE}</style></head><body>{body}</body></html>'
    out = os.path.splitext(args.md)[0] + ".html"
    open(out, "w", encoding="utf-8").write(html)
    print("wrote", out, "video" if ".mp4" in html else "gif only")


if __name__ == "__main__":
    main()
