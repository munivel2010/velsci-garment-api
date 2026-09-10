import io
import tempfile
import cv2
import numpy as np
from PIL import Image
from rembg import remove
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = FastAPI(title="Velsci Garment Pattern API")

# 1. Enable CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Velsci Garment Pattern API is Running Live!"}

@app.post("/process-garment/")
async def process_garment(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Background removal using rembg
        output_image = remove(input_image)
        img_np = np.array(output_image)
        
        if img_np.shape[2] == 4:
            alpha = img_np[:, :, 3]
            contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                
                crop_rgb = cv2.cvtColor(img_np[y:y+h, x:x+w, :3], cv2.COLOR_RGBA2RGB)
                
                # Safe temporary image creation
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                    cv2.imwrite(temp_img.name, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
                    temp_img_path = temp_img.name

                # Safe temporary PDF creation
                pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                pdf_path = pdf_file.name
                pdf_file.close()

                c_pdf = canvas.Canvas(pdf_path, pagesize=A4)
                page_w, page_h = A4
                
                c_pdf.drawString(50, page_h - 40, f"Velsci Pattern - Width: {w}px | Height: {h}px")
                c_pdf.drawImage(temp_img_path, 50, page_h - 450, width=400, preserveAspectRatio=True)
                c_pdf.showPage()
                c_pdf.save()
                
                return FileResponse(
                    pdf_path, 
                    media_type='application/pdf', 
                    filename="Velsci_Garment_Pattern.pdf"
                )

        raise HTTPException(status_code=422, detail="Garment contour could not be detected.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")
