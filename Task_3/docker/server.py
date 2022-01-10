import re
import pandas as pd
import numpy as np
import pickle
import json

from flask import Flask
from flask import request
from svd_class import SVDRecommender

app = Flask(__name__)

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'SVDRecommender':
            return SVDRecommender
        return super().find_class(module, name)

class LoadData():
    USERS = pd.read_pickle('data/pivot_table.pkl')
    SVD_MODEL = CustomUnpickler(open('models/svd.pickle.dat', 'rb')).load()

ld = LoadData()

@app.route('/predict', methods=['GET'])
def predict():
    user_id = int(request.args.get('user_id'))
    k = int(request.args.get('k'))

    user_data = ld.USERS[ld.USERS['user_id'] == user_id]
    if len(user_data) == 0 or  k <= 0:
        items = []
    else:
        items = ld.SVD_MODEL.predict(user_data, k=k)['svd'].tolist()[0]

    json_dump = json.dumps({'user_id': user_id, 'items': [int(i) for i in items]})
    return json_dump

if __name__ == '__main__':
    # app.run(port=5000, debug=True)
    app.run(port=5000, debug=True, host='0.0.0.0')