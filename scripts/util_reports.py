from pathlib import Path
from xhtml2pdf import pisa
from io import BytesIO
import os
import matplotlib.pyplot as plt
from datetime import datetime
import base64


REPORT_CSS = """
<!DOCTYPE html>
<html>
<head>
    <style>
        /* Reset and base styles */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 210mm; /* A4 width */
            margin: 0 auto;
            padding: 10mm;
            background: white;
        }
        
        /* Typography */
        h1 { 
            font-size: 24pt; 
            color: #2c3e50; 
            margin-bottom: 12pt; 
            border-bottom: 2pt solid #3498db; 
            padding-bottom: 6pt;
        }
        h2 { font-size: 18pt; color: #34495e; margin: 18pt 0 8pt; }
        h3 { font-size: 14pt; color: #2c3e50; margin: 12pt 0 6pt; }
        
        p, ul, ol { margin-bottom: 10pt; }
        strong { color: #2c3e50; }
        
        /* Tables */
        table {
            border-collapse: collapse;
            margin: 8pt 0;
            font-size: 6pt;
            max-width: 70%;
        }
        th {
            text-align: left;
            font-weight: bold;
            padding: 2pt 4pt;
            background: #f0f0f0;
            border: 1pt solid #ddd;
        }
        td {
            border: 1pt solid #e0e0e0;
            padding: 2pt 4pt;
        }
        tr:nth-child(even) { background: #f8f9fa; }
        
        /* Images */
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 2pt auto;
        }
        
        /* PDF-specific optimizations - simplified for xhtml2pdf */
        @page {
            size: A4;
            margin: 10mm;
        }
        
        /* Header styling */
        .report-header {
            text-align: center;
            margin-bottom: 24pt;
            padding-bottom: 12pt;
            border-bottom: 1pt solid #bdc3c7;
        }
        .report-title { font-size: 28pt; color: #2c3e50; }
        .report-date { 
            color: #7f8c8d; 
            font-size: 11pt; 
            margin-top: 6pt;
        }
        
        /* Footer */
        .report-footer {
            margin-top: 24pt;
            padding-top: 12pt;
            border-top: 1pt solid #bdc3c7;
            text-align: center;
            font-size: 9pt;
            color: #7f8c8d;
        }
    </style>
"""

def inject_css_into_html(html_content: str) -> str:
    """
    Inject common CSS into HTML content for consistent report styling.
    
    Args:
        html_content: Raw HTML without <html><head> tags
        
    Returns:
        Complete HTML document with CSS injected
    """
    return REPORT_CSS + html_content + "</body></html>"


def save_report_to_pdf(html_body: str, title: str, location: Path, replace: bool = False) -> Path:
    location.mkdir(parents=True, exist_ok=True)
    
    filename = f"{title}.pdf"
    pdf_path = location / filename
    
    if pdf_path.exists() and not replace:
        print("File already exists and replace flag was set to false")
        return
    
    full_page = inject_css_into_html(html_body)
    
    with open(pdf_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(full_page, dest=pdf_file)
    
    return pdf_path


def save_plot_to_png(plt_obj: plt.Figure, title: str, output_dir: Path, replace: bool = True) -> str:
    """
    Save a matplotlib plot as PNG and return relative path for HTML template.
    
    Args:
        plt_obj: Matplotlib Figure object to save
        title: Plot title used for filename (e.g., "sales_chart")
        output_dir: Path object for output directory (images will be saved here)
        replace: If True, overwrite existing file; if False, append timestamp
    
    Returns:
        Relative path string for HTML template (e.g., "images/sales_chart.png")
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    #print(plt_obj)
    # Generate filename from title
    filename = f"{title}.png"
    img_path = output_dir / filename
    
    # Handle replace logic
    if img_path.exists() and not replace:
        print("File already exists and replace flag was set to false")
        return
    
    # Save plot as PNG with high DPI for PDF quality
    plt_obj.savefig(img_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt_obj.close()  # Prevent memory leaks
    
    # Return relative path for HTML template
    return img_path

def image_to_base64(image_path: Path) -> str:
    """
    Convert image file to base64 data URI for embedding in HTML/PDF.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 data URI string (e.g., "data:image/png;base64,iVBORw0KG...")
    """
    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"