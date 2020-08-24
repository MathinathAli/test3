from flask import Flask,jsonify,request
import joblib
import os
import json
from json import JSONEncoder
import numpy

path=os.getcwd()

app = Flask(__name__)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

@app.route('/')
@app.route('/kubra')
def hello():
    no_of_bedroom=request.args.get('no_of_bedroom')
    no_of_bathrooms = request.args.get('no_of_bathrooms')
    living_area_sqft = request.args.get('living_area_sqft')
    no_of_floors = request.args.get('no_of_floors')
    rating = request.args.get('rating')
    grade = request.args.get('grade')
    top_sqft = request.args.get('top_sqft')
    basement_sqft = request.args.get('basement_sqft')
    date_of_built = request.args.get('date_of_built')
    renovated_year = request.args.get('renovated_year')
    postal_code = request.args.get('postal_code')
    latitude_coOrdinate = request.args.get('latitude_coOrdinate')
    longitude_coOrdinate = request.args.get('longitude_coOrdinate')

    sample = [[no_of_bedroom, no_of_bathrooms, living_area_sqft, no_of_floors, rating, grade, top_sqft, basement_sqft,  date_of_built, renovated_year, postal_code, latitude_coOrdinate, longitude_coOrdinate]]
    
    model=joblib.load(path+"/Model/realestate_pricepredictions.joblib")
    predicted=model.predict(sample)
    numpyData = {"array": predicted}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
    return jsonify({'data':encodedNumpyData})







