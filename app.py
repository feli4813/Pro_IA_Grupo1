from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

# Cargar el modelo entrenado
with open('modelo_energia.pkl', 'rb') as archivo:
    modelo = pickle.load(archivo)

# Cargar el encoder binarizado
with open('energia_encoder.pkl', 'rb') as archivo:
    encoder = pickle.load(archivo)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediccion', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        tipo_energia = encoder.transform([[request.form['tipo_energia']]]).toarray()
        ano = int(request.form['ano'])
        mes = int(request.form['mes'])

        # Crear un DataFrame con los datos
        datos = pd.DataFrame({
            'YEAR': [ano],
            'MONTH': [mes],
            'PRODUCT_Geothermal': [tipo_energia[0][0]],
            'PRODUCT_Hydro': [tipo_energia[0][1]],
            'PRODUCT_Solar': [tipo_energia[0][2]],
            'PRODUCT_Wind': [tipo_energia[0][3]]
        })

        # Realizar la predicción
        prediccion = (modelo.predict(datos))
        prediccion_redondeada = round(prediccion[0], 2)


        # Retornar la predicción como respuesta JSON
        return jsonify({'prediccion': prediccion_redondeada})
    except Exception as e:
        
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
