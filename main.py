import io
import os
import math
import tempfile
import cv2
import numpy as np
from PIL import Image
from rembg import remove
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from supabase import create_client, Client
import uvicorn

app = FastAPI(
    title="Velsci Garment Pattern & Cloud Storage API",
    version="1.2.0",
    description="Backend service for image pattern extraction, grid puzzle slicing, fabric estimation, and Supabase integration."
)

# CORS Policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase Credentials Setup
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://opmxgfeprpmrfkjlnqjs.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_0cOFGdJAa8FQRxuv5Ka_HQ_FCmuZUhO")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

FABRIC_SHRINKAGE = {
    "cotton": 0.08, 
    "silk": 0.02, 
    "denim": 0.05, 
    "polyester": 0.01
}

def calculate_fabric_requirement(width_px: int, height_px: int, fabric_type: str, roll_width_inch: float, price_per_meter: float) -> dict:
    scale_factor = 0.05
    actual_width_cm = width_px * scale_factor
    actual_height_cm = height_px * scale_factor
    
    shrinkage = FABRIC_SHRINKAGE.get(fabric_type.lower(), 0.05)
    
    total_height_cm = (actual_height_cm + 4.0) * (1.0 + shrinkage)
    total_width_cm = (actual_width_cm + 4.0) * (1.0 + shrinkage)
    roll_width_cm = roll_width_inch * 2.54

    if total_width_cm <= (roll_width_cm / 2):
        required_meters = (total_height_cm * 2) / 100.0
    else:
        required_meters = (total_height_cm * 2.5) / 100.0

    total_price = required_meters * price_per_meter
    return {
        "height_cm": round(total_height_cm, 1),
        "width_cm": round(total_width_cm, 1),
        "estimated_meters": round(required_meters, 2),
        "total_price": round(total_price, 2)
    }

def draw_registration_mark(c: canvas.Canvas, x: float, y: float):
    """Draws crosshair alignment target at specified coordinates."""
    c.setLineWidth(0.5)
    c.line(x - 10, y, x + 10, y)
    c.line(x, y - 10, x, y + 10)
    c.circle(x, y, 4, stroke=1, fill=0)

def generate_puzzle_pattern_pdf(image_path: str, est_data: dict, roll_id: str, fabric_type: str) -> io.BytesIO:
    """Generates a multi-page tiled PDF pattern with assembly registration marks."""
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    page_w_pt, page_h_pt = A4
    margin = 36.0  # 0.5 inch margins

    printable_w = page_w_pt - (2 * margin)
    printable_h = page_h_pt - (2 * margin)

    pt_per_cm = 28.3465
    total_w_pt = est_data["width_cm"] * pt_per_cm
    total_h_pt = est_data["height_cm"] * pt_per_cm

    cols = math.ceil(total_w_pt / printable_w)
    rows = math.ceil(total_h_pt / printable_h)

    # PAGE 1: Assembly Cover Map & Fabric Summary
    c.setFont("Helvetica-Bold", 18)
    c.drawString( margin, page_h_pt - 50, "VELSCI - DIY Garment Pattern Assembly Map")
    c.setFont("Helvetica", 11)
    c.drawString(margin, page_h_pt - 80, f"Roll ID: {roll_id} | Fabric Type: {fabric_type.upper()}")
    c.drawString(margin, page_h_pt - 100, f"Pattern Dimensions: {est_data['width_cm']} cm x {est_data['height_cm']} cm")
    c.drawString(margin, page_h_pt - 120, f"Total Fabric Required: {est_data['estimated_meters']} Meters | Cost: RS. {est_data['total_price']}")
    c.drawString(margin, page_h_pt - 140, f"Grid Structure: {rows} Rows x {cols} Columns (Total {rows * cols} Printable Sheets)")
    
    # Draw Cover Preview Image
    c.drawImage(image_path, margin, page_h_pt - 560, width=400, preserveAspectRatio=True)
    c.showPage()

    # PAGES 2+: Numbered Puzzle Tiles with Crosshair Alignment
    for r in range(rows):
        for col in range(cols):
            # Page Header
            c.setFont("Helvetica-Bold", 10)
            c.drawString(
                margin, 
                page_h_pt - 25, 
                f"VELSCI Pattern - Tile Grid [{r+1},{col+1}] (Row {r+1}/{rows}, Col {col+1}/{cols}) - Roll ID: {roll_id}"
            )

            # Draw 4 Corner Registration Crosshairs
            draw_registration_mark(c, margin, margin)
            draw_registration_mark(c, page_w_pt - margin, margin)
            draw_registration_mark(c, margin, page_h_pt - margin)
            draw_registration_mark(c, page_w_pt - margin, page_h_pt - margin)

            # Compute tile crop offset
            x_offset = -(col * printable_w) + margin
            y_offset = -(r * printable_h) + margin

            c.saveState()
            path = c.beginPath()
            path.rect(margin, margin, printable_w, printable_h)
            c.clipPath(path, stroke=0)

            c.drawImage(image_path, x_offset, y_offset, width=total_w_pt, height=total_h_pt, preserveAspectRatio=True)
            c.restoreState()

            # Tile Boundary Frame
            c.setLineWidth(0.5)
            c.rect(margin, margin, printable_w, printable_h)

            c.showPage()

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return {
        "status": "online",
        "service": "Velsci Garment Pattern & Puzzle Grid API",
        "docs_url": "/docs"
    }

@app.post("/process-and-save/", status_code=status.HTTP_201_CREATED)
async def process_and_save(
    file: UploadFile = File(...),
    fabric_type: str = Form("cotton"),
    roll_width_inch: float = Form(44.0),
    roll_id: str = Form("ROLL-DEFAULT"),
    price_per_meter: float = Form(250.0)
):
    temp_img_path = None
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Upload an image.")

        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        output_image = remove(input_image)
        img_np = np.array(output_image)
        
        if img_np.shape[2] != 4:
            raise HTTPException(status_code=422, detail="Failed to isolate image alpha channel.")

        alpha = img_np[:, :, 3]
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise HTTPException(status_code=422, detail="Garment contour could not be detected.")

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        est = calculate_fabric_requirement(w, h, fabric_type, roll_width_inch, price_per_meter)
        crop_rgb = cv2.cvtColor(img_np[y:y+h, x:x+w, :3], cv2.COLOR_RGBA2RGB)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            cv2.imwrite(temp_img.name, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
            temp_img_path = temp_img.name

        # Generate Tiled Multi-Page PDF
        pdf_buffer = generate_puzzle_pattern_pdf(temp_img_path, est, roll_id, fabric_type)

        # Upload to Supabase Storage
        file_name = f"estimates/{roll_id}_pattern.pdf"
        supabase.storage.from_('pdf_patterns').upload(
            file_name, 
            pdf_buffer.getvalue(), 
            file_options={"upsert": "true", "content-type": "application/pdf"}
        )

        pdf_public_url = supabase.storage.from_('pdf_patterns').get_public_url(file_name)

        # Save Metadata to Database
        db_data = {
            "roll_id": roll_id,
            "fabric_type": fabric_type,
            "roll_width_inch": roll_width_inch,
            "estimated_meters": est['estimated_meters'],
            "total_price": est['total_price'],
            "pdf_url": pdf_public_url
        }
        supabase.table('estimates').insert(db_data).execute()

        return {"status": "success", "data": db_data}

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)

@app.delete("/delete-estimate/{roll_id}")
async def delete_estimate(roll_id: str):
    try:
        file_name = f"estimates/{roll_id}_pattern.pdf"
        supabase.storage.from_('pdf_patterns').remove([file_name])
        supabase.table('estimates').delete().eq('roll_id', roll_id).execute()
        return {"status": "success", "message": f"Estimate for {roll_id} deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
