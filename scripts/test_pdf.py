import os
import subprocess

html_path = os.path.abspath("test_render.html")
pdf_path = os.path.abspath("test_render.pdf")

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(chrome_path):
    chrome_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(chrome_path):
    chrome_path = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

print("Using browser:", chrome_path)

cmd = [
    chrome_path,
    "--headless=new",
    "--disable-gpu",
    f"--print-to-pdf={pdf_path}",
    "--no-pdf-header-footer",
    html_path
]

res = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", res.returncode)
print("Stdout:", res.stdout)
print("Stderr:", res.stderr)
print("PDF exists:", os.path.exists(pdf_path))
if os.path.exists(pdf_path):
    print("PDF size:", os.path.getsize(pdf_path))
