"""Render report/report.md -> report/report.pdf.

Pipeline: Markdown -> HTML (tables, fenced code) -> inline figures as base64 ->
apply print CSS -> Chrome headless --print-to-pdf. Chrome is used because it is
the only PDF engine available on this machine and it renders tables/images well.
"""
from __future__ import annotations

import base64
import re
import subprocess
import sys
from pathlib import Path

import markdown  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
FIG_DIR = ROOT / "figures"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 20pt; border-bottom: 3px solid #2c3e50; padding-bottom: 4px;
     color: #1b2a3a; }
h2 { font-size: 14.5pt; margin-top: 18px; color: #1b2a3a;
     border-bottom: 1px solid #bcc6d0; padding-bottom: 2px; }
h3 { font-size: 12pt; color: #2c3e50; margin-top: 14px; }
code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px;
       font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; }
pre { background: #f6f8fa; padding: 8px 10px; border-radius: 5px; overflow-x: auto;
      border: 1px solid #e1e4e8; font-size: 8.7pt; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9pt; }
th, td { border: 1px solid #c5ccd3; padding: 4px 7px; text-align: left; }
th { background: #2c3e50; color: #fff; }
tr:nth-child(even) td { background: #f4f6f8; }
img { max-width: 100%; display: block; margin: 8px auto; }
blockquote { border-left: 4px solid #4C72B0; margin: 10px 0; padding: 4px 12px;
             background: #eef2f7; color: #2c3e50; }
.page-break { page-break-before: always; }
strong { color: #11181f; }
"""


def inline_images(html: str) -> str:
    """Rewrite every <img ... src="figures/x.png" ...> to a base64 data URI so the
    PDF is fully self-contained. Matches `src` anywhere in the tag (python-markdown
    emits `alt` before `src`)."""
    def repl(match: re.Match) -> str:
        src = match.group(1)
        path = (ROOT / src) if not Path(src).is_absolute() else Path(src)
        if not path.exists():
            path = FIG_DIR / Path(src).name
        if not path.exists():
            print(f"  [warn] missing image: {src}", file=sys.stderr)
            return match.group(0)
        b64 = base64.b64encode(path.read_bytes()).decode()
        return match.group(0).replace(f'src="{src}"', f'src="data:image/png;base64,{b64}"')
    return re.sub(r'<img[^>]*\ssrc="([^"]+)"[^>]*>', repl, html)


def main() -> None:
    md_path = REPORT_DIR / "report.md"
    pdf_path = REPORT_DIR / "report.pdf"
    html_path = REPORT_DIR / "report.html"

    body = markdown.markdown(
        md_path.read_text(),
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
    )
    body = inline_images(body)
    html = f"<!doctype html><html><head><meta charset='utf-8'>" \
           f"<style>{CSS}</style></head><body>{body}</body></html>"
    html_path.write_text(html)

    subprocess.run([
        "google-chrome", "--headless=new", "--no-sandbox",
        "--no-pdf-header-footer", "--disable-gpu",
        f"--print-to-pdf={pdf_path}", html_path.as_uri(),
    ], check=True, capture_output=True)
    size = pdf_path.stat().st_size
    print(f"Wrote {pdf_path} ({size:,} bytes)")


if __name__ == "__main__":
    main()
