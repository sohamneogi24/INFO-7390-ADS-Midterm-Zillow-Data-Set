from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt, cos, sin

import gpxpy.geo

app = Flask(__name__)

@app.route('/test', methods = ['GET'])
def getTest():
    return jsonify({'parcelid' : 1})

@app.route('/predict', methods = ['POST'])
def getPredictions() :
    request_dict = request.json['predictions']
    parcelid = request_dict.pop('parcelid')
    prediction = process_request(request_dict)
    return jsonify({'parcelid' : parcelid, 'prediction-value' : prediction})

def process_request(request_data) :
    return load_model_and_predict(request_data)

def load_model_and_predict(request_data) :
    df = pd.DataFrame(list(request_data.items()))
    new_df = df.transpose()
    header = new_df.iloc[0]
    request_test_data = new_df[1:]
    request_test_data.columns = header

    rf_model = joblib.load('./model/random_forest_model.pkl')
    prediction = rf_model.predict(request_test_data)
    print("*********** PREDICTION *************", prediction)
    return prediction[0]


@app.route('/search/<string:latitude>/<string:longitude>', methods = ['GET'])
def geoSpatialSearch(latitude, longitude) :
    return jsonify({'results' : search(latitude=float(latitude )/ 10e5,longitude = float(longitude)/10e5,dist=0.1 )})


def search(latitude, longitude, dist) :


    bounding_lon_1 = longitude - (dist / (abs(cos(latitude) * 69)))
    bounding_lon_2 = longitude + (dist / (abs(cos(latitude) * 69)))

    bounding_lat_1 = latitude - (dist / 69)
    bounding_lat_2 = latitude + (dist / 69)

    stripped_df = pd.read_csv('./data/search.csv')
    filtered_df = stripped_df[(stripped_df['latitude'] / 10e5 > bounding_lat_1) & (stripped_df['latitude'] / 10e5 < bounding_lat_2) & (stripped_df['longitude'] / 10e5 > bounding_lon_1) & (stripped_df['longitude'] / 10e5 < bounding_lon_2)]
    filtered_df['distance'] = filtered_df.apply(
        lambda x: gpxpy.geo.haversine_distance(latitude, longitude, x['latitude'] / 10e5, x['longitude'] / 10e5) / 1000, axis=1)
    top_10 = filtered_df.sort_values(by='distance', ascending=True).head(10)
    if(top_10.shape[0] == 10) :
        return top_10.to_dict(orient='records')
    else :
        return search(latitude=latitude, longitude=longitude, dist= dist * 2)




if __name__ == '__main__':
    app.run(debug= True, port = 8000, host ='0.0.0.0')
