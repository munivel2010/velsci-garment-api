from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import cv2
import numpy as np
import io
from PIL import Image
from rembg import remove
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile

app = FastAPI(title="Velsci Garment Pattern API")

@app.get("/")
def home():
    return {"message": "Velsci Garment Pattern API is Running Live!"}

@app.post("/process-garment/")
async def process_garment(file: UploadFile = File(...)):
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents))
    
    # rembg மூலம் பின்னணியை நீக்குதல்
    output_image = remove(input_image)
    img_np = np.array(output_image)
    
    if img_np.shape[2] == 4:
        alpha = img_np[:, :, 3]
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            
            crop_rgb = cv2.cvtColor(img_np[y:y+h, x:x+w, :3], cv2.COLOR_RGBA2RGB)
            
            temp_img_path = tempfile.mktemp(suffix=".png")
            cv2.imwrite(temp_img_path, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
            
            # Printable PDF உருவாக்குதல்
            pdf_path = tempfile.mktemp(suffix=".pdf")
            c_pdf = canvas.Canvas(pdf_path, pagesize=A4)
            page_w, page_h = A4
            
            c_pdf.drawString(50, page_h - 40, f"Velsci Pattern - Width: {w}px | Height: {h}px")
            c_pdf.drawImage(temp_img_path, 50, page_h - 450, width=400, preserveAspectRatio=True)
            c_pdf.showPage()
            c_pdf.save()
            
            return FileResponse(pdf_path, media_type='application/pdf', filename="Velsci_Garment_Pattern.pdf")

    return {"error": "துணியைக் கண்டறிய முடியவில்லை."}
