# from flask import Flask, request, jsonify
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import pickle
#
# # Загружаем обученную модель
# with open('output/linear_model.pkl', 'rb') as f:
#     lr_model = pickle.load(f)
#
# # Создаем приложение Flask
# app = Flask(__name__)
#
# # API для предсказания цены
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Получение данных из запроса
#         input_data = request.json
#         df = pd.DataFrame([input_data])
#
#         # Прогнозирование
#         prediction = lr_model.predict(df)
#         return jsonify({'predicted_price': prediction[0]})
#     except Exception as e:
#         return jsonify({'error': str(e)})
#
# # Запуск приложения
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Загружаем модель
with open("output/linear_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return "API is running!"
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)