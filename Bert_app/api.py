import uvicorn
from fastapi import FastAPI
import model


app = FastAPI()


@app.get('/')
def index():
	return {"Summarify Bert Model Deployment" : "Modeli kullanmak için /docs adresine gidin"}

# ML

@app.post('/predict/{text}')
def predict(text: str):
	input_id, attention_mask = model.tokenize(text)
	prediction = model.predict(input_id, attention_mask)
	return {"prediction":prediction}


if __name__ == '__main__':
	uvicorn.run(app,host="127.0.0.1", port=8000)