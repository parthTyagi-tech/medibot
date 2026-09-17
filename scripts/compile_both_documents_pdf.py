"""
Script to compile both MediAssist markdown documents into high-resolution,
executive-grade PDFs using Headless Chrome/Edge, and save copies directly
to D:\\Downloads.

Documents:
1. docs/MediAssist_Technical_Masterclass.md -> D:\\Downloads\\MediAssist_Technical_Masterclass.pdf
2. docs/study_notes_and_hld.md -> D:\\Downloads\\MediAssist_Study_Notes_and_HLD.pdf
"""

import os
import re
import shutil
import base64
import subprocess
from markdown_it import MarkdownIt

DOCS_DIR = os.path.abspath("docs")
DOWNLOADS_DIR = r"D:\Downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

DOCUMENTS_TO_COMPILE = [
    {
        "title": "MediAssist Technical Masterclass",
        "md_path": os.path.join(DOCS_DIR, "MediAssist_Technical_Masterclass.md"),
        "pdf_name": "MediAssist_Technical_Masterclass.pdf"
    },
    {
        "title": "MediAssist Study Notes & HLD Architecture",
        "md_path": os.path.join(DOCS_DIR, "study_notes_and_hld.md"),
        "pdf_name": "MediAssist_Study_Notes_and_HLD.pdf"
    }
]

def find_browser():
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

def get_whitepaper_css(doc_title):
    return f"""
    @page {{
        size: A4;
        margin: 16mm 14mm 16mm 14mm;
    }}
    @page :left {{
        @bottom-left {{ content: "{doc_title}"; font-size: 8pt; color: #64748b; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        @bottom-right {{ content: counter(page); font-size: 8pt; color: #64748b; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    }}
    @page :right {{
        @bottom-left {{ content: "MediAssist AI Engineering Architecture"; font-size: 8pt; color: #64748b; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        @bottom-right {{ content: counter(page); font-size: 8pt; color: #64748b; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    }}

    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #1e293b;
        background-color: #ffffff;
        font-size: 9.6pt;
        line-height: 1.55;
        margin: 0;
        padding: 0;
    }}

    h1 {{
        color: #0f172a;
        font-size: 19pt;
        font-weight: 800;
        border-bottom: 2.5px solid #0284c7;
        padding-bottom: 6px;
        margin-top: 0;
        margin-bottom: 12px;
        page-break-after: avoid;
        break-after: avoid;
    }}

    h2 {{
        color: #0f172a;
        font-size: 13.5pt;
        font-weight: 700;
        border-bottom: 1.5px solid #e2e8f0;
        padding-bottom: 4px;
        margin-top: 20px;
        margin-bottom: 8px;
        page-break-after: avoid;
        break-after: avoid;
    }}

    h3 {{
        color: #0369a1;
        font-size: 11pt;
        font-weight: 700;
        margin-top: 14px;
        margin-bottom: 5px;
        page-break-after: avoid;
        break-after: avoid;
    }}

    h4 {{
        color: #334155;
        font-size: 10pt;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 4px;
    }}

    p {{
        margin-top: 0;
        margin-bottom: 8px;
        text-align: justify;
    }}

    code {{
        font-family: "Consolas", "Courier New", monospace;
        font-size: 8.8pt;
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 1.5px 4px;
        border-radius: 3px;
        border: 1px solid #e2e8f0;
    }}

    pre {{
        background-color: #0f172a;
        color: #f8fafc;
        padding: 10px 12px;
        border-radius: 6px;
        font-size: 8pt;
        line-height: 1.4;
        overflow-x: auto;
        page-break-inside: avoid;
        break-inside: avoid;
        margin-top: 6px;
        margin-bottom: 12px;
    }}

    pre code {{
        background-color: transparent;
        color: #f8fafc;
        padding: 0;
        border: none;
        font-size: 8pt;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 8px;
        margin-bottom: 12px;
        font-size: 8.8pt;
        page-break-inside: avoid;
        break-inside: avoid;
    }}

    th, td {{
        border: 1px solid #cbd5e1;
        padding: 6px 8px;
        text-align: left;
    }}

    th {{
        background-color: #f8fafc;
        color: #0f172a;
        font-weight: 700;
    }}

    tr:nth-child(even) {{
        background-color: #f8fafc;
    }}

    img {{
        max-width: 100%;
        height: auto;
        display: block;
        margin: 12px auto;
        border-radius: 6px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        page-break-inside: avoid;
        break-inside: avoid;
    }}

    blockquote {{
        border-left: 3.5px solid #0284c7;
        background-color: #f0f9ff;
        color: #0369a1;
        margin: 8px 0;
        padding: 6px 12px;
        border-radius: 0 4px 4px 0;
        font-size: 9.2pt;
    }}

    ul, ol {{
        margin-top: 2px;
        margin-bottom: 8px;
        padding-left: 20px;
    }}

    li {{
        margin-bottom: 3px;
    }}

    hr {{
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 16px 0;
    }}
    """

def compile_document(doc_info, browser_bin):
    md_path = doc_info["md_path"]
    pdf_name = doc_info["pdf_name"]
    title = doc_info["title"]
    
    local_pdf = os.path.join(DOCS_DIR, pdf_name)
    downloads_pdf = os.path.join(DOWNLOADS_DIR, pdf_name)
    temp_html = os.path.join(DOCS_DIR, f"temp_{os.path.splitext(pdf_name)[0]}.html")

    print(f"\n========================================================")
    print(f"Processing: {title}")
    print(f"Source: {md_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    md = MarkdownIt("default").enable("table").enable("strikethrough")
    raw_html = md.render(md_text)

    # Convert relative images to Base64
    html_with_images = convert_images_to_base64(raw_html, DOCS_DIR)
    css = get_whitepaper_css(title)

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
{html_with_images}
</body>
</html>"""

    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Rendering PDF via Headless Chrome...")
    cmd = [
        browser_bin,
        "--headless=new",
        "--disable-gpu",
        f"--print-to-pdf={local_pdf}",
        "--no-pdf-header-footer",
        temp_html
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Browser error stdout:", res.stdout)
        print("Browser error stderr:", res.stderr)
        raise RuntimeError(f"Browser failed with exit code {res.returncode}")

    if os.path.exists(temp_html):
        os.remove(temp_html)

    if os.path.exists(local_pdf):
        # Copy to D:\Downloads
        shutil.copyfile(local_pdf, downloads_pdf)
        size_kb = os.path.getsize(downloads_pdf) / 1024
        print(f"[SUCCESS] Generated & Saved to D:\\Downloads:")
        print(f"   -> {downloads_pdf} ({size_kb:.1f} KB)")
    else:
        raise FileNotFoundError(f"PDF was not generated at {local_pdf}")

def main():
    browser_bin = find_browser()
    print(f"Found Browser: {browser_bin}")
    for doc in DOCUMENTS_TO_COMPILE:
        compile_document(doc, browser_bin)
    print("\n========================================================")
    print(f"[SUCCESS] All documents compiled and successfully saved to {DOWNLOADS_DIR}!")

if __name__ == "__main__":
    main()
