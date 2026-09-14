import math
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def generate_tiled_puzzle_pdf(image_path, output_pdf, image_width_cm, image_height_cm):
    """
    Slices a large pattern image across an N x M grid of A4 pages with
    alignment registration marks and numbered grid labels.
    """
    page_w_pt, page_h_pt = A4  # 1 pt = 1/72 inch (~595 x 842 pt for A4)
    margin = 36  # 0.5 inch printable margin
    
    printable_w = page_w_pt - (2 * margin)
    printable_h = page_h_pt - (2 * margin)

    # Convert pattern dimensions to points (1 cm ≈ 28.3465 points)
    pt_per_cm = 28.3465
    total_w_pt = image_width_cm * pt_per_cm
    total_h_pt = image_height_cm * pt_per_cm

    # Calculate required grid columns and rows
    cols = math.ceil(total_w_pt / printable_w)
    rows = math.ceil(total_h_pt / printable_h)

    c = canvas.Canvas(output_pdf, pagesize=A4)

    for r in range(rows):
        for col in range(cols):
            # Page Title / Puzzle Identification
            c.setFont("Helvetica-Bold", 10)
            c.drawString(margin, page_h_pt - 25, f"VELSCI Pattern Puzzle - Grid [{r+1},{col+1}] (Row {r+1}, Col {col+1} of {rows}x{cols})")

            # Registration Crosshairs (Alignment Targets) at Page Corners
            draw_registration_mark(c, margin, margin)
            draw_registration_mark(c, page_w_pt - margin, margin)
            draw_registration_mark(c, margin, page_h_pt - margin)
            draw_registration_mark(c, page_w_pt - margin, page_h_pt - margin)

            # Calculate crop area for this grid tile
            x_offset = -(col * printable_w) + margin
            y_offset = -(r * printable_h) + margin

            c.saveState()
            # Clip drawing area strictly within printable margins
            path = c.beginPath()
            path.rect(margin, margin, printable_w, printable_h)
            c.clipPath(path, stroke=0)

            # Draw the scaled image offset for the current tile
            c.drawImage(image_path, x_offset, y_offset, width=total_w_pt, height=total_h_pt, preserveAspectRatio=True)
            c.restoreState()

            # Draw border around printable tile area
            c.setLineWidth(0.5)
            c.rect(margin, margin, printable_w, printable_h)

            c.showPage()

    c.save()

def draw_registration_mark(c, x, y):
    """Draws an alignment crosshair target at (x, y) coordinates."""
    c.setLineWidth(0.5)
    c.line(x - 10, y, x + 10, y)
    c.line(x, y - 10, x, y + 10)
    c.circle(x, y, 4, stroke=1, fill=0)
