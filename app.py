from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os
from preprocessing import TextPreprocessor

app = FastAPI(title="Phishing Detection API")

models = {}
tokenizers = {}
preprocessor = TextPreprocessor()
device = torch.device('cpu')

class PredictionRequest(BaseModel):
    text: str
    data_type: str

class PredictionResponse(BaseModel):
    is_phishing: bool
    confidence: float
    label: str
    data_type: str

@app.on_event("startup")
async def load_models():
    print("Loading models...")
    for model_type in ['sms', 'email']:
        model_path = f'./roberta-{model_type}'
        if os.path.exists(model_path):
            try:
                tokenizers[model_type] = RobertaTokenizer.from_pretrained(model_path)
                models[model_type] = RobertaForSequenceClassification.from_pretrained(model_path)
                models[model_type].to(device)
                models[model_type].eval()
                print(f"Loaded {model_type} model")
            except Exception as e:
                print(f"Error loading {model_type}: {e}")
    print("Models loaded")

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Phishing Detection</title>
    <style>
        body {font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px;}
        h1 {text-align: center;}
        select, textarea, button {width: 100%; padding: 10px; margin: 10px 0; font-size: 16px;}
        textarea {min-height: 150px; font-family: monospace;}
        button {background: #007bff; color: white; border: none; cursor: pointer; border-radius: 5px;}
        button:hover {background: #0056b3;}
        .result {margin-top: 20px; padding: 20px; border-radius: 10px; display: none;}
        .result.show {display: block;}
        .phishing {background: #f8d7da; border-left: 5px solid #dc3545;}
        .safe {background: #d4edda; border-left: 5px solid #28a745;}
    </style>
</head>
<body>
    <h1>Phishing Detection API</h1>
    <p>RoBERTa Transformer for SMS and Email</p>
    
    <select id="dataType">
        <option value="sms">SMS</option>
        <option value="email">Email</option>
    </select>
    
    <textarea id="textInput" placeholder="Enter text..."></textarea>
    <button onclick="analyze()">Analyze</button>
    
    <div class="result" id="result"></div>
    
    <script>
        async function analyze() {
            const text = document.getElementById('textInput').value;
            const dataType = document.getElementById('dataType').value;
            
            if (!text) {
                alert('Please enter text');
                return;
            }
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text, data_type: dataType})
                });
                
                const data = await response.json();
                const resultDiv = document.getElementById('result');
                resultDiv.className = 'result show ' + (data.is_phishing ? 'phishing' : 'safe');
                resultDiv.innerHTML = '<h2>' + data.label + '</h2><p>Confidence: ' + (data.confidence * 100).toFixed(1) + '%</p>';
            } catch (error) {
                alert('Error: ' + error);
            }
        }
    </script>
</body>
</html>
    """

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": list(models.keys())}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if request.data_type not in models:
        raise HTTPException(status_code=503, detail=f"Model for {request.data_type} not available")
    
    try:
        cleaned_text = preprocessor.clean_text(request.text)
        model = models[request.data_type]
        tokenizer = tokenizers[request.data_type]
        
        inputs = tokenizer(cleaned_text, return_tensors='pt', truncation=True, max_length=128, padding='max_length').to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs).item()
            confidence = probs[0][prediction].item()
        
        return PredictionResponse(
            is_phishing=bool(prediction == 1),
            confidence=float(confidence),
            label="PHISHING" if prediction == 1 else "SAFE",
            data_type=request.data_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    print("Starting API on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
