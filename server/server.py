from flask import Flask
from flask import request
import json

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


def ml_function(r): return r


@app.route("/chatbot", methods=['GET', 'POST'])
def chatbot():
    payload = request.data
    magic = ml_function(payload)
    return magic
