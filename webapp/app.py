
import requests
from flask import Flask
from flask import request, url_for
from flask import jsonify, make_response
from recommender import Recommender

import pandas as pd 
import numpy as np 
from pandas import DataFrame

data_path = './static/'

CHATFUEL_URL = "chatfuel_api"
FB_API_URL = 'https://graph.facebook.com/v2.6/me/messages'
VERIFY_TOKEN = 'test_token'
PAGE_ACCESS_TOKEN = 'EAAdrJJDrZAqIBADZAl6EEWTPR3BBoQ5sjuSPnwb1fzb4gZA2zg4zByCdlZAXvK5mh02XQarSCpVZAq3cC5YlZB4klXWxdTZB2kAmT0KqpynxZAd8ZCqrTS89s1MOzQz1ZCIfXaQeBxjDaMa3UNMVGRfS5je3yb7AHE5BRmmOfevpRvoejE9YCSjJnK'

# Initialize Flask app.	
app = Flask(__name__)

# Initialize the recommender object.
recommender=Recommender()

def get_bot_response(message):
	return "this is a dummy response to '{}'".format(message)


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

def send_message(recipient_id, text):
	""" send response to FB """
	# log("sending message to {recipient}: {text}".format(recipient=recipient_id, text=text))

	headers = {
	"Content-Type": "application/json"
	}
	payload = {
	"message": {
		"text": text
	},
	"recipient": {
	"id": recipient_id
	},
	"notification_type": "regular"
	}

	auth = {
		'access_token': PAGE_ACCESS_TOKEN
	}

	response = requests.post(
		FB_API_URL,
		params=auth,
		headers=headers,
		json=payload
		)
	if response.status_code != 200:
		log(response.status_code)
		log(response.text)

	return response


def get_recommendation_index(user_data):
	# init recommender and the weights at random
	rr = Recommender(n_fixed_feature=len(user_data))
	rr.init_weights()

	""" need to add a user_goal and user_rating function?? """

	user_goal = [['time', 0.5], ['talk', 0.5],
				['friendly', 0.5], ['advice', 0.5]]
	user_rating = {'Group Therapy': (3, 5),'Vent Over Tea': (5, 5),'7 Cups': (4, 5)}

	# get the service index of top_k choices
	choices, service_names = rr.get_recommendation(
		user_profile, user_goal, verbose=False)
	choices_for_chatbot = [c + 1 for c in choices]

	user_scores = rr.process_user_rating(user_rating)
	rr.update_weights(user_profile, user_scores, choices, verbose=False)

	return choices_for_chatbot


@app.route("/webhook", methods=['GET','POST'])
def listen():
	""" main flask function for listening at webhook endpoint """
	if request.method == 'GET':
		return verify_webhook(request)

	if request.method == 'POST':
		payload = request.json
		# event = payload['entry'][0]['messaging']
		# for x in event:
		# 	if is_user_message(x):
		# 		text = x['message']['text']
		# 		sender_id = x['sender']['id']
		# 		respond(sender_id, text)

		return "Hello World", 200


@app.route("/testing", methods=['GET','POST'])
def json():
	""" returns a message to messenger if json file successfully received """
	if request.json:
		mydata = request.json
		text = "thanks i got it!"
		return make_response(jsonify({
			"messages": [
			{"text": text
			}
			]
			}))
	else:
		return make_response(jsonify({
			"messages": [
			{"text": "no json received"
			}
			]
			}))


@app.route("/results", methods=['POST', 'GET'])
def show_results():


	if request.method == 'POST':
		received = request.json
		service_url = data_path + 'services.csv'
		service_data = pd.read_csv(service_url)

		""" retrieve info for best matching """
		descriptions = service_data['full_desc']
		websites = service_data['website']
		names = service_data['name']

		reco1 = received['reco1']
		desc_1 = descriptions[reco1]
		website_1 = websites[reco1]
		name_1 = names[reco1]

		reco2 = received['reco2']
		desc_2 = descriptions[reco2]
		website_2 = websites[reco2]
		name_2 = names[reco2]

		reco3 = received['reco3']
		desc_3 = descriptions[reco3]
		website_3 = websites[reco3]
		name_3 = names[reco3]

		return make_response(jsonify({
			"messages": [
			{
				"attachment": {
					"type": "template",
						"payload": {
						"template_type": "button",
						"text": desc_1,
						"buttons": [
							{
				"type": "web_url",
				"url": website_1,
				"title": name_1
							}
				]
						}
					}},
			{
				"attachment": {
					"type": "template",
						"payload": {
						"template_type": "button",
						"text": desc_2,
						"buttons": [
							{
				"type": "web_url",
				"url": website_2,
				"title": name_2
							}
				]
						}
					}},
			{
				"attachment": {
					"type": "template",
						"payload": {
						"template_type": "button",
						"text": desc_3,
						"buttons": [
							{
				"type": "web_url",
				"url": website_3,
				"title": name_3
							}
				]
						}
					}}
				]
			}))





@app.route("/rec", methods=['GET', 'POST'])
def recommendation():
	""" will take the json file and push it through the python model """
	recommendation = [0,0,0]
	if request.json:
		# a Python dictionary with JSON fields serielized into key/value pairs
		data = request.json
		# feed data into the recommender to get get reco(mmendation
		recommendation = get_recommendation_index(data)
		# return a JSON response
		return make_response(jsonify({
			# "messages": [
			# {
			# "text": "Your recommendations are {}, {}, and {}.".format(name, name, name)
			# }]
			# {
			"set_attributes": {
				"reco1": recommendation[0],
				"reco2": recommendation[1],
				"reco3": recommendation[2]
				}
			}))
