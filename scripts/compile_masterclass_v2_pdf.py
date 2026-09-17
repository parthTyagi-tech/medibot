"""
Script to compile docs/MediAssist_Technical_Masterclass_v2.md into a high-quality PDF
using Headless Chrome with executive-grade typography, responsive diagrams,
and print-optimized CSS pagination. Also copies the PDF to D:\Downloads.
"""

import os
import re
import shutil
import base64
import subprocess
from markdown_it import MarkdownIt

DOCS_DIR = os.path.abspath("docs")
MD_PATH = os.path.join(DOCS_DIR, "MediAssist_Technical_Masterclass_v2.md")
PDF_PATH = os.path.join(DOCS_DIR, "MediAssist_Technical_Masterclass_v2.pdf")
DOWNLOADS_DIR = r"D:\Downloads"
DOWNLOADS_PDF = os.path.join(DOWNLOADS_DIR, "MediAssist_Technical_Masterclass_v2.pdf")
TEMP_HTML = os.path.join(DOCS_DIR, "temp_masterclass_v2_render.html")

def find_chrome():
    paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Neither Google Chrome nor Microsoft Edge was found on this system.")

def convert_images_to_base64(html_content, base_dir):
    """Replace relative image src with embedded base64 data URIs for 100% reliable headless rendering."""
    def replace_img(match):
        src = match.group(1)
        if src.startswith("./"):
            local_path = os.path.normpath(os.path.join(base_dir, src[2:]))
        elif not os.path.isabs(src):
            local_path = os.path.normpath(os.path.join(base_dir, src))
        else:
            local_path = src
        
        if os.path.exists(local_path):
            ext = os.path.splitext(local_path)[1].lower().replace(".", "")
            mime = f"image/{ext}" if ext != "svg" else "image/svg+xml"
            with open(local_path, "rb") as img_f:
                b64_data = base64.b64encode(img_f.read()).decode("utf-8")
            return f'<img src="data:{mime};base64,{b64_data}"'
        return match.group(0)

    return re.sub(r'<img\s+src=["\']([^"\']+)["\']', replace_img, html_content)

def build_pdf():
    print(f"Reading Markdown: {MD_PATH}")
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    md = MarkdownIt("default").enable("table").enable("strikethrough")
    raw_html = md.render(md_text)

    # Convert relative images to Base64
    html_with_images = convert_images_to_base64(raw_html, DOCS_DIR)

    # Custom Whitepaper CSS
    css = """
    @page {
        size: A4;
        margin: 18mm 16mm 18mm 16mm;
    }
    @page :left {
        @bottom-left { content: "MediAssist v2 Distributed AI Masterclass"; font-size: 8pt; color: #64748b; font-family: sans-serif; }
        @bottom-right { content: counter(page); font-size: 8pt; color: #64748b; font-family: sans-serif; }
    }
    @page :right {
        @bottom-left { content: "Principal AI Infrastructure Architecture"; font-size: 8pt; color: #64748b; font-family: sans-serif; }
        @bottom-right { content: counter(page); font-size: 8pt; color: #64748b; font-family: sans-serif; }
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #1e293b;
        background-color: #ffffff;
        font-size: 9.8pt;
        line-height: 1.55;
        margin: 0;
        padding: 0;
    }

    h1 {
        color: #0f172a;
        font-size: 20pt;
        font-weight: 800;
        border-bottom: 2.5px solid #0284c7;
        padding-bottom: 6px;
        margin-top: 0;
        margin-bottom: 14px;
        page-break-after: avoid;
        break-after: avoid;
    }

    h2 {
        color: #0f172a;
        font-size: 14pt;
        font-weight: 700;
        border-bottom: 1.5px solid #e2e8f0;
        padding-bottom: 4px;
        margin-top: 22px;
        margin-bottom: 10px;
        page-break-after: avoid;
        break-after: avoid;
    }

    h3 {
        color: #0369a1;
        font-size: 11.5pt;
        font-weight: 700;
        margin-top: 16px;
        margin-bottom: 6px;
        page-break-after: avoid;
        break-after: avoid;
    }

    h4 {
        color: #334155;
        font-size: 10.2pt;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 4px;
        page-break-after: avoid;
        break-after: avoid;
    }

    p, li {
        color: #334155;
        text-align: justify;
    }

    strong {
        color: #0f172a;
    }

    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 20px 0;
    }

    /* Blockquotes as Executive Callout Cards */
    blockquote {
        background-color: #f8fafc;
        border-left: 4px solid #0284c7;
        margin: 12px 0;
        padding: 8px 14px;
        color: #334155;
        border-radius: 0 6px 6px 0;
        page-break-inside: avoid;
        break-inside: avoid;
    }

    blockquote p {
        margin: 4px 0;
    }

    /* Tables */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
        font-size: 8.8pt;
        page-break-inside: avoid;
        break-inside: avoid;
    }

    th {
        background-color: #0f172a;
        color: #ffffff;
        font-weight: 600;
        text-align: left;
        padding: 7px 10px;
        border: 1px solid #1e293b;
    }

    td {
        padding: 6px 10px;
        border: 1px solid #e2e8f0;
        vertical-align: top;
    }

    tr:nth-child(even) td {
        background-color: #f8fafc;
    }

    /* Code Blocks */
    pre {
        background-color: #0f172a;
        color: #e2e8f0;
        border-radius: 6px;
        padding: 10px 14px;
        font-family: "Consolas", "Courier New", monospace;
        font-size: 8.2pt;
        line-height: 1.45;
        overflow-x: auto;
        margin: 12px 0;
        page-break-inside: avoid;
        break-inside: avoid;
        border: 1px solid #1e293b;
    }

    code {
        font-family: "Consolas", "Courier New", monospace;
        font-size: 8.5pt;
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 1px 4px;
        border-radius: 4px;
    }

    pre code {
        background-color: transparent;
        color: #e2e8f0;
        padding: 0;
    }

    /* Images */
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 14px auto;
        border-radius: 6px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
        page-break-inside: avoid;
        break-inside: avoid;
    }

    ul, ol {
        padding-left: 20px;
        margin-top: 4px;
        margin-bottom: 10px;
    }

    li {
        margin-bottom: 3px;
    }
    """

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MediAssist v2 Distributed AI Infrastructure Masterclass</title>
<style>
{css}
</style>
</head>
<body>
{html_with_images}
</body>
</html>"""

    print(f"Writing temporary HTML: {TEMP_HTML}")
    with open(TEMP_HTML, "w", encoding="utf-8") as f:
        f.write(full_html)

    browser_bin = find_chrome()
    print(f"Rendering PDF with browser: {browser_bin}")

    cmd = [
        browser_bin,
        "--headless=new",
        "--disable-gpu",
        f"--print-to-pdf={PDF_PATH}",
        "--no-pdf-header-footer",
        TEMP_HTML
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Browser error stdout:", res.stdout)
        print("Browser error stderr:", res.stderr)
        raise RuntimeError(f"Browser failed with exit code {res.returncode}")

    if os.path.exists(TEMP_HTML):
        os.remove(TEMP_HTML)

    if os.path.exists(PDF_PATH):
        size_kb = os.path.getsize(PDF_PATH) / 1024
        print(f"Success! Masterclass v2 PDF compiled: {PDF_PATH} ({size_kb:.1f} KB)")
        
        # Copy to D:\Downloads
        os.makedirs(DOWNLOADS_DIR, exist_ok=True)
        shutil.copy2(PDF_PATH, DOWNLOADS_PDF)
        print(f"Copied to: {DOWNLOADS_PDF} ({os.path.getsize(DOWNLOADS_PDF)/1024:.1f} KB)")
    else:
        raise FileNotFoundError(f"PDF was not generated at {PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
