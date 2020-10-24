import uvicorn
from fastapi import FastAPI
import distilbert_model_ktrain as model


app = FastAPI()


@app.get('/')
def index():
	return {"Summarify Bert Model Deployment" : "Modeli kullanmak için /docs adresine gidin"}

# ML

@app.post('/predict/{text}')
def predict(text: str):
	prediction = model.get_prediction(text)
	return {"prediction":prediction}


if __name__ == '__main__':
	uvicorn.run(app,host="127.0.0.1", port=8000)