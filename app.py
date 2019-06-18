
import requests
from flask import Flask
from flask import request
from flask import jsonify, make_response
from fakeRecommender import Recommender

FB_API_URL = 'https://graph.facebook.com/v2.6/me/messages'
VERIFY_TOKEN = 'test_token'
PAGE_ACCESS_TOKEN = 'EAAdrJJDrZAqIBADZAl6EEWTPR3BBoQ5sjuSPnwb1fzb4gZA2zg4zByCdlZAXvK5mh02XQarSCpVZAq3cC5YlZB4klXWxdTZB2kAmT0KqpynxZAd8ZCqrTS89s1MOzQz1ZCIfXaQeBxjDaMa3UNMVGRfS5je3yb7AHE5BRmmOfevpRvoejE9YCSjJnK'

# Initialize Flask app.	
app = Flask(__name__)

# Initialize the recommender object.
recommender=Recommender()

def get_bot_response(message):
	return "this is a dummy response to '{}'.format(message)"


def verify_webhook(req):
	""" Check page token is correct """
	if req.args.get("hub.verify_token") == VERIFY_TOKEN:
		return req.args.get("hub.challenge")
	else: 
		return "incorrect"

def respond(sender, message):
	""" makes a response to the user and passes it to a function that sends it"""
	response = get_bot_response(message)
	send_message(sender, response)

def is_user_message(message):
	""" checks if message is from user """
	return (message.get('message') and not message['message'].get("is_echo"))

@app.route("/webhook")

def listen():
	""" main flask function for listening at webhook endpoint """
	if request.method == 'GET':
		return verify_webhook(request)

	if request.method == 'POST':
		payload = request.json
		event = payload['entry'][0]['messaging']
		for x in event:
			if is_user_message(x):
				test = x['message']['text']
				sender_id = x['sender']['id']
				respond(sender_id, text)

		return "ok"

def send_message(recipient_id, text):
	""" send response to FB """
	payload = {
	'message': {
		'text': text
	},
	'recipient': {
	'id': recipient_id
	},
	'notification_type': 'regular'
	}

	auth = {
		'access_token': PAGE_ACCESS_TOKEN
	}

	response = requests.post(
		FB_API_URL,
		params=auth,
		json=payload
		)

	return response.json

@app.route("/testing")

def testing():
	if request.method == 'GET':
		print(request)
		return request

	if request.method == 'POST':
		payload = request.json
		return request

# This route will receive and parse some JSON data and then return a JSON response using the recommender
@app.route("/rec", methods=['GET', 'POST'])
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