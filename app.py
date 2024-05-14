from fastapi import FastAPI
from classification import Cbullying_Classification
from pydantic import BaseModel
#to run api open terminal and write uvicorn app:app 
#to test api open postman and put the link in the termnal add /classify/ then body -> raw and put the in body {"text": "ur text here"}

app = FastAPI()
cb=Cbullying_Classification("facebook/bart-large-mnli")

class TextData(BaseModel):
    text: str
    threshold: float

@app.post("/classify/")
def classify_text_api(text_data: TextData):
    classification = cb.classify(text_data.text,text_data.threshold)
    return classification