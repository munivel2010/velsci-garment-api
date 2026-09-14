import io
import os
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
    version="1.1.0",
    description="Backend service for image pattern extraction, fabric estimation, and Supabase integration."
)

# CORS Policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production setup: specify exact domains e.g., ["https://velsci.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase Credentials Setup
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://opmxgfeprpmrfkjlnqjs.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_KEY:
    # Fallback to key provided in environment configuration
    SUPABASE_KEY = "sb_publishable_0cOFGdJAa8FQRxuv5Ka_HQ_FCmuZUhO"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Shrinkage rates per fabric type
FABRIC_SHRINKAGE = {
    "cotton": 0.08, 
    "silk": 0.02, 
    "denim": 0.05, 
    "polyester": 0.01
}

def calculate_fabric_requirement(width_px: int, height_px: int, fabric_type: str, roll_width_inch: float, price_per_meter: float) -> dict:
    scale_factor = 0.05  # Scale px to cm approximation
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

@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return {
        "status": "online",
        "service": "Velsci Garment Pattern API",
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
        # Validate Upload File Type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Background Removal
        output_image = remove(input_image)
        img_np = np.array(output_image)
        
        if img_np.shape[2] != 4:
            raise HTTPException(status_code=422, detail="Failed to process image alpha channel.")

        alpha = img_np[:, :, 3]
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise HTTPException(status_code=422, detail="Garment contour could not be detected.")

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        est = calculate_fabric_requirement(w, h, fabric_type, roll_width_inch, price_per_meter)
        crop_rgb = cv2.cvtColor(img_np[y:y+h, x:x+w, :3], cv2.COLOR_RGBA2RGB)
        
        # Save temporary image for PDF rendering
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            cv2.imwrite(temp_img.name, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
            temp_img_path = temp_img.name

        # Generate PDF in-memory using BytesIO buffer
        pdf_buffer = io.BytesIO()
        c_pdf = canvas.Canvas(pdf_buffer, pagesize=A4)
        page_w, page_h = A4
        
        c_pdf.setFont("Helvetica-Bold", 16)
        c_pdf.drawString(50, page_h - 40, "VELSCI - Garment Pattern & Fabric Estimate")
        c_pdf.setFont("Helvetica", 10)
        c_pdf.drawString(50, page_h - 65, f"Roll ID: {roll_id} | Fabric Type: {fabric_type.upper()}")
        c_pdf.drawString(50, page_h - 80, f"Required Fabric: {est['estimated_meters']} Meters | Cost: RS. {est['total_price']}")
        c_pdf.drawImage(temp_img_path, 50, page_h - 520, width=400, preserveAspectRatio=True)
        c_pdf.showPage()
        c_pdf.save()

        pdf_buffer.seek(0)

        # Upload PDF to Supabase Storage
        file_name = f"estimates/{roll_id}_pattern.pdf"
        supabase.storage.from_('pdf_patterns').upload(
            file_name, 
            pdf_buffer.getvalue(), 
            file_options={"upsert": "true", "content-type": "application/pdf"}
        )

        pdf_public_url = supabase.storage.from_('pdf_patterns').get_public_url(file_name)

        # Save Metadata to Supabase Database
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
        # Cleanup temporary files from local disk
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)

@app.delete("/delete-estimate/{roll_id}")
async def delete_estimate(roll_id: str):
    try:
        file_name = f"estimates/{roll_id}_pattern.pdf"
        
        # 1. Remove PDF from Supabase storage
        supabase.storage.from_('pdf_patterns').remove([file_name])

        # 2. Delete row from Database
        supabase.table('estimates').delete().eq('roll_id', roll_id).execute()

        return {"status": "success", "message": f"Estimate for {roll_id} deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
