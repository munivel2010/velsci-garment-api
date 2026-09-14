import pytest
from fastapi.testclient import TestClient
from main import app, calculate_fabric_requirement

client = TestClient(app)

# 1. Test calculation logic independently
def test_calculate_fabric_requirement():
    result = calculate_fabric_requirement(
        width_px=500, 
        height_px=1000, 
        fabric_type="cotton", 
        roll_width_inch=44.0, 
        price_per_meter=200.0
    )
    assert "estimated_meters" in result
    assert result["estimated_meters"] > 0
    assert "total_price" in result

# 2. Test full API pattern creation endpoint
def test_process_and_save_endpoint(tmp_path):
    # Create a dummy image file for testing
    import cv2
    import numpy as np
    
    img_path = tmp_path / "test_garment.png"
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(dummy_img, (150, 150), 80, (255, 255, 255), -1)  # Draw shape
    cv2.imwrite(str(img_path), dummy_img)

    with open(img_path, "rb") as f:
        response = client.post(
            "/process-and-save/",
            files={"file": ("test_garment.png", f, "image/png")},
            data={
                "fabric_type": "cotton",
                "roll_width_inch": "44.0",
                "roll_id": "PYTEST-TEST-01",
                "price_per_meter": "300.0"
            }
        )
    
    assert response.status_code == 201
    json_data = response.json()
    assert json_data["status"] == "success"
    assert "pdf_url" in json_data["data"]
