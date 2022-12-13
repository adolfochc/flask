# Primero, cargamos el modelo usando pickle
import pickle
from flask import request,jsonify
from flask import Flask
import numpy as np
import pandas as pd
import os
from flask_cors import CORS


with open('decision_tree_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
with open("linear_regression_model.pkl", "rb") as f:
  model_lineal = pickle.load(f)

# Luego, creamos la función que se encargará de hacer predicciones
# con el modelo y exponerla como una API

app = Flask(__name__)

CORS(app)
# this is the entry point
#application = app

@app.route("/")
def hello():
    return "Hello World!"


@app.route('/predictv2', methods=['POST'])
def predictv2():
    # Obtenemos los datos de entrada de la solicitud HTTP
    input_data = request.json
    #
    # Crear un diccionario con los datos
    datos = {'Mantenimiento': [164], 'TipoVehiculo': [26],'Equipo':[45],'Flota':[8],'CodigoMat':[2709],'Material':[6675],'KmOT':[841409],'(CORRECTIVO,)':0,
         '(PREVENTIVO,)':1,'(PROACTIVO,)':0,'(SINIESTROS,)':0,'(TRABAJOS MAYORES - MEJORAS,)':0,'(TRABAJOS MENORES - IMAGEN,)':0}
    # Crear la tabla a partir del diccionario
    tabla = pd.DataFrame(datos)
    # Convertimos el diccionario a una matriz NumPy
    #input_array = np.array(list(input_data.values()))
    # Aquí deberías procesar los datos de entrada y usar el modelo
    # para hacer una predicción
    prediction = model_lineal.predict(tabla)

    # Finalmente, devolvemos la predicción en formato JSON
    #return jsonify(prediction)
    #value = np.datetime64(prediction,'s')
    #value  = prediction.astype(np.datetime64)
    value = prediction[0].astype(np.int64)
    int64_series = pd.Series([value])
    datetime_series = pd.to_datetime(int64_series,format='%d/%m/%Y')
    
    # convert the Series to a JSON string
    json_string = datetime_series.to_json()
    # print the JSON string
    print(json_string)
    print(value)
    #print(np.datetime64(int(value),'s'))
    print(datetime_series[0])
    response = jsonify(
        {
            'data': datetime_series[0]
        }
    )
    return response

@app.route('/predict', methods=['POST'])
def predict():
    # Obtenemos los datos de entrada de la solicitud HTTP
    input_data = request.json
    # Convertimos el diccionario a una matriz NumPy
    input_array = np.array(list(input_data.values()))
    # Aquí deberías procesar los datos de entrada y usar el modelo
    # para hacer una predicción
    prediction = model.predict(input_array)

    # Finalmente, devolvemos la predicción en formato JSON
    #return jsonify(prediction)
    #value = np.datetime64(prediction,'s')
    #value  = prediction.astype(np.datetime64)
    value = prediction[0].astype(np.int64)
    int64_series = pd.Series([value])
    datetime_series = pd.to_datetime(int64_series)
    # convert the Series to a JSON string
    json_string = datetime_series.to_json()
    # print the JSON string
    print(json_string)
    print(value)
    #print(np.datetime64(int(value),'s'))
    print(datetime_series[0])
    response = jsonify(
        {
            'data': datetime_series[0]
        }
    )
    return response

#if __name__ == '__main__':
    #app.run()
    
if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True, port=os.getenv("PORT", default=8000))


#pip install numpy
#pip install pandas
#pip install scikit-learn
