from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import logging
import time
import os
from datetime import datetime
import pickle

# Configure logging
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_filename = os.path.join(log_directory, f"app_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


filename = 'language_detection_pipeline.pkl'

logger.info(f"Loading language detection model from {filename}")
try:
    with open(filename, 'rb') as f:
        loaded_pipeline = pickle.load(f)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Prediction input class
class TextInput(BaseModel):
    text: str = Field(..., examples=["This is a sample text in English.",
                                     "Questo è un testo di esempio in italiano.",
                                     "C'est un texte d'exemple en français.",])

# Prediction output class
class PredictionOutput(BaseModel):
    language_code: str = Field(..., description="Predicted language code")
    confidence: float = Field(..., description="Confidence score of the prediction", ge=0.0, le=1.0)

# Create FastAPI app
app = FastAPI(title="Language Detection API")

# Add request logging using FastAPI's built-in middleware decorator
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    # Log request
    request_id = f"{time.time()}"
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Request {request_id} started - Method: {request.method} Path: {request.url.path} Client: {client_host}")

    # Process request
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # Log response
    logger.info(f"Request {request_id} completed - Status: {response.status_code} Process time: {process_time:.4f}s")

    return response

@app.post("/predict-language")
async def predict_language(input_text: TextInput) -> PredictionOutput:
    """
    Predict the language of the input test.
    """
    # Log the incoming prediction request
    logger.info(f"Received language prediction request with text: {input_text.text[:50]}{'...' if len(input_text.text) > 50 else ''}")

    # Get the text from the input
    text = input_text.text

    # Make the prediction
    logger.info("Making language prediction")
    prediction = loaded_pipeline.predict([text])
    confidence = loaded_pipeline.predict_proba([text]).max()
    logger.info(f"Prediction result: language={prediction[0]}, confidence={confidence:.4f}")

    # Create the output
    output = PredictionOutput(
        language_code=prediction[0],
        confidence=confidence
    )

    logger.info(f"Returning prediction: {output.language_code} with confidence {output.confidence:.4f}")
    return output

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)