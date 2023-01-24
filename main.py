# Primero, cargamos el modelo usando pickle
import pickle
from flask import request,jsonify
from flask import Flask
import numpy as np
import pandas as pd
import os
from flask_cors import CORS
import random
import datetime


with open('decision_tree_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
with open("linear_regression_model.pkl", "rb") as f:
  model_lineal = pickle.load(f)

with open("arbol_regression.pkl", "rb") as fv:
  model_regression = pickle.load(fv)

# Luego, creamos la función que se encargará de hacer predicciones
# con el modelo y exponerla como una API

app = Flask(__name__)

CORS(app)
# this is the entry point
#application = app

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/predictv3', methods=['POST'])
def predictv3():
    # Obtenemos los datos de entrada de la solicitud HTTP
    input_data = request.json
    #print("REQUEST => ",input_data)
    # Leer la tabla externa
    tabla_conversion_equipo = pd.read_excel("Equipo.xlsx")
    tabla_conversion_material = pd.read_excel("Material.xlsx")
    # Crear un diccionario con los datos
    #datos = {'KmOT':[input_data['kmot']],'Fecha Prog.': [input_data['fecha']]}
    datos = {'Equipo': input_data['equipo'],'Material':input_data['material'],'KmOT':input_data['kmot']}
    # Crear la tabla a partir del diccionario
    #tabla = pd.DataFrame(datos)
    #tabla = pd.DataFrame(datos, columns=['KmOT','Fecha Prog.'])
    tabla = pd.DataFrame(datos, columns=['Equipo','Material','KmOT'])
    #print(tabla.info())
    #print(tabla)
    #print("---------------------------------")
    #print(tabla_conversion_equipo.info())
    # Unir la tabla externa con el DataFrame original en base a la columna "valor" solo con las coincidencias y dejando solo la columna "id"
    tabla = pd.merge(tabla, tabla_conversion_equipo[["valor", "id"]], left_on='Equipo', right_on="valor", how='inner').drop(columns=['valor'])
    tabla = tabla.drop(['Equipo'], axis=1)
    tabla = tabla.rename(columns={"id": "Equipo"})
    #print(tabla.info())
    tabla = pd.merge(tabla, tabla_conversion_material[["valor", "id"]], left_on='Material', right_on="valor", how='inner').drop(columns=['valor'])
    tabla = tabla.drop(['Material'], axis=1)
    tabla = tabla.rename(columns={"id": "Material"})
    print(tabla.info())
    tabla = tabla[['Equipo','Material','KmOT']]
    print(tabla.info())

    #tabla['Fecha Prog.'] = tabla['Fecha Prog.'].astype(np.datetime64)
    #tabla['Fecha Prog.'] = tabla['Fecha Prog.'].astype(np.int64)
    
    # Generar un número aleatorio de días entre 1 y 30 (inclusive)
    #random_days = random.randint(30, 90)
    
    #print(tabla)
    predictions = model_regression.predict(tabla)
    array_pred = np.array(predictions)
    int64_series = pd.Series(array_pred)
    datetime_series = pd.to_datetime(int64_series, unit='ns')
    #int64_series = pd.Series([predictions[0]])
    #datetime_series = pd.to_datetime(int64_series)
    print("Predicted:", datetime_series[0])
    #fecha_convertida = datetime_series[0].strftime("%Y-%m-%d %H:%M:%S")
    fecha_p = datetime_series[0]
    # Sumar el número aleatorio de días al objeto datetime
    #datetime_object_future = fecha_p +  datetime.timedelta(days=random_days)

    #fecha_convertida = datetime_object_future.strftime("%Y-%m-%d %H:%M:%S")
    #fecha_convertida = "error"
    response_pred = datetime_series.dt.strftime('%Y-%m-%d %H:%M:%S')
    print(response_pred)
    response_pred_dict = response_pred.to_dict()
    data_list = [{"fecha_predic": val} for val in response_pred_dict.values()]

    #data_list = response_pred.tolist()
    response = jsonify(
        {
            'data': data_list
        }
    )
    return response


@app.route('/predictv2', methods=['POST'])
def predictv2():
    # Obtenemos los datos de entrada de la solicitud HTTP
    input_data = request.json
    print("REQUEST => ",input_data)
    # Crear un diccionario con los datos
    #datos = {'Mantenimiento': [160], 'TipoVehiculo': [24],'Equipo':[5],'Flota':[8],'CodigoMat':[2709],'Material':[6675],'KmOT':[841409],'(CORRECTIVO,)':0,
     #    '(PREVENTIVO,)':1,'(PROACTIVO,)':0,'(SINIESTROS,)':0,'(TRABAJOS MAYORES - MEJORAS,)':0,'(TRABAJOS MENORES - IMAGEN,)':0}
    datos = {'Fecha Prog.': [160],'KmOT':[841409]}
    # Crear la tabla a partir del diccionario
    tabla = pd.DataFrame(datos)
    # Convertimos el diccionario a una matriz NumPy
    #input_array = np.array(list(input_data.values()))
    # Aquí deberías procesar los datos de entrada y usar el modelo
    # para hacer una predicción
    prediction = model_lineal_v2.predict(tabla)

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
    fecha_convertida = datetime_series[0].strftime("%Y-%m-%d")
    response = jsonify(
        {
            'data': fecha_convertida
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
    #app.run(host="0.0.0.0",debug=True, port=os.getenv("PORT", default=3000))
    app.run(host="0.0.0.0",debug=True, port=os.getenv("PORT", default=8000))


#pip install numpy
#pip install pandas
#pip install scikit-learn
