from fastapi import FastAPI
from pydantic import BaseModel, Field

import pickle

filename = 'language_detection_pipeline.pkl'
loaded_pipeline = pickle.load(open(filename, 'rb'))

# Text input model - list of strings
text_to_predict = ["Questo è un testo di esempio in italiano."]

# Prediction
predicted_language = loaded_pipeline.predict(text_to_predict)

print(f"Predicted language: {predicted_language[0]}")

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

app = FastAPI()

@app.post("/predict-language")
async def predict_language(input_text: TextInput) -> PredictionOutput:
    """
    Predict the language of the input test.
    """

    # Get the text from the input
    text = input_text.text

    # Make the prediction
    prediction = loaded_pipeline.predict([text])
    confidence = loaded_pipeline.predict_proba([text]).max()

    # Create the output
    output = PredictionOutput(
        language_code=prediction[0],
        confidence=confidence
    )

    return output

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)