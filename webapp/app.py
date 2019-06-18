
from flask import Flask
from flask import request
from flask import jsonify, make_response
from fakeRecommender import Recommender

# Initialize Flask app.	
app = Flask(__name__)

# Initialize the recommender object.
recommender=Recommender()

# This route will receive and parse some JSON data and then return a JSON response using the recommender
@app.route("/chatbot", methods=['GET', 'POST'])
def chatbot():
	recommendation = "default: conselling"
	if request.method == 'POST':
		# a Python dictionary with JSON fields serielized into key/value pairs
		data=request.get_json(force=True)
		# feed data into the recommender to get get recommendation
		recommendation = recommender.recommend(data)
		# return a JSON response
		return make_response(jsonify({"recommendation":recommendation}))
	else:
		return make_response(jsonify({"recommendation":recommendation}))