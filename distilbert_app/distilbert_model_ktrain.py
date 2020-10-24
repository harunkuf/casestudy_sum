import ktrain

predictor = ktrain.load_predictor("distilbert")

def get_prediction(x):
	pred = predictor.predict([x])
	return pred[0]