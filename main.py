import io
import tempfile
import cv2
import numpy as np
from PIL import Image
from rembg import remove
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import uvicorn

app = FastAPI(title="Velsci Garment Pattern & Smart Fabric Estimator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. துணி வகைகளின் சுருங்கும் தன்மை (Shrinkage Rates)
FABRIC_SHRINKAGE = {
    "cotton": 0.08,   # 8% சுருங்கும்
    "silk": 0.02,     # 2% சுருங்கும்
    "denim": 0.05,    # 5% சுருங்கும்
    "polyester": 0.01 # 1% சுருங்கும்
}

def calculate_fabric_requirement(
    width_px: int, 
    height_px: int, 
    fabric_type: str, 
    roll_width_inch: float,
    price_per_meter: float = 250.0,
    scale_factor: float = 0.05
):
    actual_width_cm = width_px * scale_factor
    actual_height_cm = height_px * scale_factor

    # Fabric Shrinkage 적용
    shrinkage = FABRIC_SHRINKAGE.get(fabric_type.lower(), 0.05)
    
    # Seam Allowance (4cm) + Shrinkage Factor
    total_height_cm = (actual_height_cm + 4.0) * (1.0 + shrinkage)
    total_width_cm = (actual_width_cm + 4.0) * (1.0 + shrinkage)

    # Convert Roll Width Inches to CM (1 inch = 2.54 cm)
    roll_width_cm = roll_width_inch * 2.54

    # Pattern Nesting Logic
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

@app.post("/process-garment/")
async def process_garment(
    file: UploadFile = File(...),
    fabric_type: str = Form("cotton"),
    roll_width_inch: float = Form(44.0),
    roll_id: str = Form("ROLL-DEFAULT"),
    price_per_meter: float = Form(250.0)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        output_image = remove(input_image)
        img_np = np.array(output_image)
        
        if img_np.shape[2] == 4:
            alpha = img_np[:, :, 3]
            contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                
                # Smart Fabric & Price Estimate Calculation
                est = calculate_fabric_requirement(
                    w, h, fabric_type, roll_width_inch, price_per_meter
                )

                crop_rgb = cv2.cvtColor(img_np[y:y+h, x:x+w, :3], cv2.COLOR_RGBA2RGB)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                    cv2.imwrite(temp_img.name, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
                    temp_img_path = temp_img.name

                pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                pdf_path = pdf_file.name
                pdf_file.close()

                c_pdf = canvas.Canvas(pdf_path, pagesize=A4)
                page_w, page_h = A4
                
                # PDF Printable Header & Material Details
                c_pdf.setFont("Helvetica-Bold", 16)
                c_pdf.drawString(50, page_h - 40, "VELSCI - Garment Pattern & Fabric Estimate")
                
                c_pdf.setFont("Helvetica", 10)
                c_pdf.drawString(50, page_h - 65, f"Roll ID: {roll_id} | Fabric Type: {fabric_type.upper()}")
                c_pdf.drawString(50, page_h - 80, f"Roll Width: {roll_width_inch}\" | Pattern Size: {est['width_cm']}cm x {est['height_cm']}cm")
                
                c_pdf.setFont("Helvetica-Bold", 11)
                c_pdf.drawString(50, page_h - 100, f"Required Fabric: {est['estimated_meters']} Meters")
                c_pdf.drawString(50, page_h - 115, f"Estimated Material Cost: RS. {est['total_price']}")

                c_pdf.drawImage(temp_img_path, 50, page_h - 520, width=400, preserveAspectRatio=True)
                c_pdf.showPage()
                c_pdf.save()
                
                return FileResponse(
                    pdf_path, 
                    media_type='application/pdf', 
                    filename=f"Velsci_Estimate_{roll_id}.pdf"
                )

        raise HTTPException(status_code=422, detail="Garment contour could not be detected.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
