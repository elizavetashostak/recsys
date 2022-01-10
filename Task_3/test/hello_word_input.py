from flask import Flask
from flask import request
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def hello_world_input():
    data = request.query_string
    return 'Hello, World!' + str(data)