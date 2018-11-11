from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge
import pickle
from sklearn.externals import joblib
from flask import Flask

app = Flask(__name__)

transformer = pickle.load(open("transformer.p", "rb"))
model = joblib.load('model_bayesian.joblib')

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        crime_rate = json_data['crime_rate']
        avg_number_of_rooms = json_data['avg_number_of_rooms']
        distance_to_employment_centers = json_data['distance_to_employment_centers']
        property_tax_rate = json_data['property_tax_rate']
        pupil_teacher_ratio = json_data['pupil_teacher_ratio']
        if crime_rate is None or avg_number_of_rooms is None \
                or distance_to_employment_centers is None \
                or property_tax_rate is None or pupil_teacher_ratio is None:
            return 0
        else:
            names_col = ["crime_rate", "avg_number_of_rooms", "distance_to_employment_centers",
                         "property_tax_rate", "pupil_teacher_ratio"]
            input_data = pd.DataFrame([[crime_rate,avg_number_of_rooms,distance_to_employment_centers,property_tax_rate,pupil_teacher_ratio]], columns=names_col)
        if isinstance(input_data, pd.core.series.Series):
            input_data = input_data.to_frame().transpose()
        
        res = model.predict(transformer.transform(input_data[names_col]), return_std=True)

        return jsonify(house_value = res[0].tolist(), stddev = res[1].tolist())

api.add_resource(HelloWorld, '/predict')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5444,debug=True)
