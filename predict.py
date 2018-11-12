from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge
import pickle
from sklearn.externals import joblib
from flask import Flask

transformer = pickle.load(open("transformer.p", "rb"))
model = joblib.load('model_bayesian.joblib')

app = Flask(__name__)
api = Api(app)

@app.route('/my400')
def my400(msg = None):
    code = 400
    if msg is None:
        msg = '400 Bad request'
    else:
        msg = '400 Bad request - ' + msg
    return msg, code

class HelloWorld(Resource):
    
    def post(self):
        json_data = request.get_json(force=True)
        json_names = list(json_data.keys())
        if 'crime_rate' not in json_names or 'avg_number_of_rooms' not in json_names or 'distance_to_employment_centers' not in json_names or 'property_tax_rate' not in json_names or 'pupil_teacher_ratio' not in json_names:
            return my400()
        crime_rate = json_data['crime_rate']        
        avg_number_of_rooms = json_data['avg_number_of_rooms']
        distance_to_employment_centers = json_data['distance_to_employment_centers']
        property_tax_rate = json_data['property_tax_rate']
        pupil_teacher_ratio = json_data['pupil_teacher_ratio']
        if crime_rate > 1.0 or crime_rate < 0:
            return my400("crime_rate should be between 0 and 1")        
        elif avg_number_of_rooms > 11 or avg_number_of_rooms < 0:
            return my400("avg_number_of_rooms should be between 0 and 11")
        elif distance_to_employment_centers < 0 or distance_to_employment_centers > 15:
            return my400("distance_to_employment_centers should be between 0 and 15")
        elif property_tax_rate < 100 or property_tax_rate > 1000:
            return my400("property_tax_rate should be between 100 and 1000")
        elif pupil_teacher_ratio < 0 or pupil_teacher_ratio > 30:
            return my400("pupil_teacher_ratio should be between 0 and 30")
        names_col = ["crime_rate", "avg_number_of_rooms", "distance_to_employment_centers",
                         "property_tax_rate", "pupil_teacher_ratio"]
        input_data = pd.DataFrame([[crime_rate,avg_number_of_rooms,distance_to_employment_centers,property_tax_rate,pupil_teacher_ratio]], columns=names_col)
            
        if isinstance(input_data, pd.core.series.Series):
            input_data = input_data.to_frame().transpose()
        
        res = model.predict(transformer.transform(input_data[names_col]), return_std=True)

        return jsonify(house_value = res[0].tolist(), stddev = res[1].tolist())

api.add_resource(HelloWorld, '/predict')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
