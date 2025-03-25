from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import cv2
import numpy as np
from roboflow import Roboflow
import supervision as sv
import uvicorn
import os
import google.generativeai as genai
from typing import Optional
from pydantic import BaseModel

# Model for chat request/response
class ChatRequest(BaseModel):
    message: str
    
class ChatResponse(BaseModel):
    reply: str

# Initialize FastAPI app
app = FastAPI(title="Urban Mobility API")

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Roboflow for vehicle detection
rf = Roboflow(api_key="pjZtbsAzjhkBsKvruel1")
project = rf.workspace().project("traffic_management-fmnmk")
model = project.version(5).model

# Initialize Google Gemini AI
genai.configure(api_key="AIzaSyCH5foXWnw35EWPs9PHOStSRwt6rb-bD5I")

# Create the Gemini model with configuration
generation_config = {
    "temperature": 0.7,  # Slightly lower than your original config for more consistent responses
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 4096,  # Reduced from 8192 for faster responses
    "response_mime_type": "text/plain",
}

# Initialize the model
gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Store chat sessions with a simple dictionary
# In production, use a database or Redis for persistence
chat_sessions = {}

def get_chat_session(session_id: str = "default"):
    """Get or create a chat session"""
    if session_id not in chat_sessions:
        # Initialize with urban mobility context
        chat_sessions[session_id] = gemini_model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": ["You are an urban mobility and traffic assistant. You provide helpful information about traffic management, urban transportation, road safety, and related topics. Keep responses concise and focused on urban mobility topics."]
                },
                {
                    "role": "model", 
                    "parts": ["I'm your urban mobility and traffic assistant. I can help with information about traffic management, transportation systems, road safety, urban planning, and related topics. How can I assist you with your urban mobility questions today?"]
                }
            ]
        )
    return chat_sessions[session_id]

@app.post("/detect/")
async def detect_vehicles(file: UploadFile = File(...)):
    """Detect vehicles in an uploaded image"""
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Save temporary image for Roboflow (required by the API)
    temp_image_path = "temp_image.jpg"
    cv2.imwrite(temp_image_path, image)
    
    # Make prediction using Roboflow
    try:
        result = model.predict(temp_image_path, confidence=40, overlap=30).json()
        
        # Get labels and detections
        labels = [item["class"] for item in result["predictions"]]
        detections = sv.Detections.from_inference(result)
        
        # Annotate image
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )
        
        # Count detections by class
        detection_counts = {}
        for label in labels:
            if label in detection_counts:
                detection_counts[label] += 1
            else:
                detection_counts[label] = 1
        
        # Convert image to bytes for response
        is_success, buffer = cv2.imencode(".jpg", annotated_image)
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode image")
        
        image_bytes = io.BytesIO(buffer)
        image_bytes.seek(0)
        
        # Return annotated image as streaming response
        return {
            "total_detections": len(labels),
            "detection_counts": detection_counts,
            "image_url": f"/image/{file.filename}"  # URL to access the image
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/image/{filename}")
async def get_image(file: UploadFile = File(...)):
    """Return processed image with annotations"""
    # Read image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process image with Roboflow model
    temp_image_path = "temp_image.jpg"
    cv2.imwrite(temp_image_path, image)
    
    result = model.predict(temp_image_path, confidence=40, overlap=30).json()
    labels = [item["class"] for item in result["predictions"]]
    detections = sv.Detections.from_inference(result)
    # Using BoxAnnotator instead of MaskAnnotator since masks might not be available
    box_annotator = sv.BoxAnnotator()
    # Use TOP_LEFT position which doesn't require masks
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

    # Annotate with boxes first
    annotated_image = box_annotator.annotate(
        scene=image, detections=detections)
    # Then add labels
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)
    # Annotate image
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels
    )
    
    # Convert image to bytes for response
    is_success, buffer = cv2.imencode(".jpg", annotated_image)
    image_bytes = io.BytesIO(buffer)
    image_bytes.seek(0)
    
    return StreamingResponse(image_bytes, media_type="image/jpeg")

@app.post("/chat/", response_model=ChatResponse)
async def chat_with_ai(
    message: str = Form(...),
    session_id: Optional[str] = Form("default")
):
    """Chat with the AI about urban mobility topics"""
    try:
        # Get or create chat session
        chat_session = get_chat_session(session_id)
        
        # Send message to Gemini
        response = chat_session.send_message(message)
        
        # Return the response text
        return ChatResponse(reply=response.text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/analyze-traffic/")
async def analyze_traffic(
    file: UploadFile = File(...),
    query: str = Form(...)
):
    """Analyze traffic image and respond to a query about it"""
    try:
        # First detect vehicles
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Save temporary image for Roboflow
        temp_image_path = "temp_image.jpg"
        cv2.imwrite(temp_image_path, image)
        
        # Get detection results
        result = model.predict(temp_image_path, confidence=40, overlap=30).json()
        
        # Extract key information
        labels = [item["class"] for item in result["predictions"]]
        
        # Count detections by class
        detection_counts = {}
        for label in labels:
            if label in detection_counts:
                detection_counts[label] += 1
            else:
                detection_counts[label] = 1
        
        # Create a summary of the detection
        detection_summary = f"I analyzed the traffic image and found {len(labels)} vehicles: "
        detection_summary += ", ".join([f"{count} {vehicle_type}" for vehicle_type, count in detection_counts.items()])
        
        # Send the detection summary and original query to Gemini
        chat_session = get_chat_session("analysis")
        prompt = f"Based on this traffic analysis: {detection_summary}\n\nUser query: {query}\n\nProvide a helpful response about the traffic situation."
        
        response = chat_session.send_message(prompt)
        
        return {
            "detection_results": {
                "total_detections": len(labels),
                "detection_counts": detection_counts
            },
            "analysis": response.text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing traffic: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Urban Mobility API is running",
        "endpoints": {
            "detect": "POST /detect/ - Upload an image to detect vehicles",
            "chat": "POST /chat/ - Chat with AI about urban mobility topics",
            "analyze-traffic": "POST /analyze-traffic/ - Upload an image and ask a question about it"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)