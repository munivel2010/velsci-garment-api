import io
import os
import math
import json
import base64
import tempfile
import requests
import cv2
import numpy as np
from PIL import Image
from rembg import remove
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from supabase import create_client, Client
from openai import OpenAI
import uvicorn

app = FastAPI(
    title="Velsci Garment Pattern & AI Design API",
    version="2.0.0",
    description="AI-powered backend for garment restyling, seam landmark detection, fabric estimation, and printable grid PDF generation."
)

# CORS Policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment Variables Setup
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://opmxgfeprpmrfkjlnqjs.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_0cOFGdJAa8FQRxuv5Ka_HQ_FCmuZUhO")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

FABRIC_SHRINKAGE = {"cotton": 0.08, "silk": 0.02, "denim": 0.05, "polyester": 0.01}

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
    c.setLineWidth(0.5)
    c.line(x - 10, y, x + 10, y)
    c.line(x, y - 10, x, y + 10)
    c.circle(x, y, 4, stroke=1, fill=0)

def generate_puzzle_pattern_pdf(image_path: str, est_data: dict, roll_id: str, fabric_type: str, ai_landmarks: dict = None) -> io.BytesIO:
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    page_w_pt, page_h_pt = A4
    margin = 36.0

    printable_w = page_w_pt - (2 * margin)
    printable_h = page_h_pt - (2 * margin)

    pt_per_cm = 28.3465
    total_w_pt = est_data["width_cm"] * pt_per_cm
    total_h_pt = est_data["height_cm"] * pt_per_cm

    cols = math.ceil(total_w_pt / printable_w)
    rows = math.ceil(total_h_pt / printable_h)

    # Cover Page
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, page_h_pt - 50, "VELSCI - AI Garment Pattern & Assembly Map")
    c.setFont("Helvetica", 10)
    c.drawString(margin, page_h_pt - 75, f"Roll ID: {roll_id} | Fabric Type: {fabric_type.upper()}")
    c.drawString(margin, page_h_pt - 90, f"Pattern Dimensions: {est_data['width_cm']} cm x {est_data['height_cm']} cm")
    c.drawString(margin, page_h_pt - 105, f"Required Fabric: {est_data['estimated_meters']} Meters | Est. Cost: RS. {est_data['total_price']}")
    c.drawString(margin, page_h_pt - 120, f"Grid Puzzle: {rows} Rows x {cols} Columns (Total {rows * cols} Sheets)")

    if ai_landmarks:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, page_h_pt - 145, "AI Seam Analysis:")
        c.setFont("Helvetica", 9)
        c.drawString(margin, page_h_pt - 160, f"Garment Category: {ai_landmarks.get('garment_type', 'Generic')}")
        c.drawString(margin, page_h_pt - 175, f"Identified Panels: {', '.join(ai_landmarks.get('panels', []))}")

    c.drawImage(image_path, margin, page_h_pt - 580, width=380, preserveAspectRatio=True)
    c.showPage()

    # Tile Pages
    for r in range(rows):
        for col in range(cols):
            c.setFont("Helvetica-Bold", 10)
            c.drawString(margin, page_h_pt - 25, f"VELSCI Tile [{r+1},{col+1}] (Row {r+1}/{rows}, Col {col+1}/{cols}) - ID: {roll_id}")

            draw_registration_mark(c, margin, margin)
            draw_registration_mark(c, page_w_pt - margin, margin)
            draw_registration_mark(c, margin, page_h_pt - margin)
            draw_registration_mark(c, page_w_pt - margin, page_h_pt - margin)

            x_offset = -(col * printable_w) + margin
            y_offset = -(r * printable_h) + margin

            c.saveState()
            path = c.beginPath()
            path.rect(margin, margin, printable_w, printable_h)
            c.clipPath(path, stroke=0)

            c.drawImage(image_path, x_offset, y_offset, width=total_w_pt, height=total_h_pt, preserveAspectRatio=True)
            c.restoreState()

            c.setLineWidth(0.5)
            c.rect(margin, margin, printable_w, printable_h)
            c.showPage()

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

@app.get("/")
async def root():
    return {"status": "online", "service": "Velsci AI Garment & Pattern API", "docs_url": "/docs"}

# 1. AI Restyling Endpoint (Generates new garment photo based on selected style)
@app.post("/generate-style/")
async def generate_style(
    file: UploadFile = File(...),
    style_prompt: str = Form("Convert this garment into a V-neck sleeveless summer dress with ruffled hem")
):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable is not configured.")

    try:
        # Read uploaded image bytes
        contents = await file.read()
        
        # Call DALL-E 3 image generation API based on the restyling prompt
        prompt = f"A clear, full-body 2D flat lay fashion illustration of a garment restyled as follows: {style_prompt}. Minimal studio background, high detail, suitable for sewing pattern extraction."
        
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )

        generated_image_url = response.data[0].url
        return {
            "status": "success",
            "style_prompt": style_prompt,
            "generated_image_url": generated_image_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Style generation failed: {str(e)}")

# 2. AI Pattern Extraction & Tiling Endpoint
@app.post("/process-and-save-ai/", status_code=status.HTTP_201_CREATED)
async def process_and_save_ai(
    file: UploadFile = File(...),
    fabric_type: str = Form("cotton"),
    roll_width_inch: float = Form(44.0),
    roll_id: str = Form("ROLL-AI-01"),
    price_per_meter: float = Form(250.0)
):
    temp_img_path = None
    try:
        contents = await file.read()
        
        # GPT-4o Vision Seam Analysis
        ai_landmarks = {}
        if openai_client:
            try:
                base64_image = base64.b64encode(contents).decode('utf-8')
                vision_response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text", 
                                    "text": "Analyze this garment for sewing pattern extraction. Return JSON with keys: 'garment_type' (str) and 'panels' (list of strings e.g. ['front_body', 'sleeve_left', 'collar'])."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                }
                            ]
                        }
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=300
                )
                ai_landmarks = json.loads(vision_response.choices[0].message.content)
            except Exception:
                ai_landmarks = {"garment_type": "Detected Garment", "panels": ["front_panel", "back_panel"]}

        # OpenCV Contour Extraction
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        output_image = remove(input_image)
        img_np = np.array(output_image)
        
        if img_np.shape[2] != 4:
            raise HTTPException(status_code=422, detail="Alpha extraction failed.")

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

        # Generate Multi-Page PDF with AI Insights
        pdf_buffer = generate_puzzle_pattern_pdf(temp_img_path, est, roll_id, fabric_type, ai_landmarks)

        # Upload to Supabase Storage
        file_name = f"estimates/{roll_id}_pattern.pdf"
        supabase.storage.from_('pdf_patterns').upload(
            file_name, 
            pdf_buffer.getvalue(), 
            file_options={"upsert": "true", "content-type": "application/pdf"}
        )

        pdf_public_url = supabase.storage.from_('pdf_patterns').get_public_url(file_name)

        db_data = {
            "roll_id": roll_id,
            "fabric_type": fabric_type,
            "roll_width_inch": roll_width_inch,
            "estimated_meters": est['estimated_meters'],
            "total_price": est['total_price'],
            "pdf_url": pdf_public_url
        }
        supabase.table('estimates').insert(db_data).execute()

        return {"status": "success", "ai_analysis": ai_landmarks, "data": db_data}

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
