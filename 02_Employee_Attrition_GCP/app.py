import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import joblib

app = Flask(__name__)
model = joblib.load('hr_analytics_model.sav')


@app.route('/')
def home():
    # return 'Hello World'
    return render_template('home.html')
    # return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    # The monthly hours , years spent in the organisation was caled during ML model training
    # Have to scale in the same range
    # Monthly hours (96 to 310)
    # years spent (2 to 10)
    monthly_hours_min = 96
    monthly_hours_max = 310
    time_spend_min = 2
    time_spend_max = 10
    user_average_monthly_hours = int_features[3]
    user_time_spend_company = int_features[4]

    scaled_average_monthly_hours = (user_average_monthly_hours - monthly_hours_min) / (
            monthly_hours_max - monthly_hours_min
    )
    int_features[3] = scaled_average_monthly_hours

    scaled_time_spend = (user_time_spend_company - time_spend_min) / (time_spend_max - time_spend_min)
    int_features[4] = scaled_time_spend

    final_features = [np.array(int_features)]
    pred = model.predict(final_features)

    # Predict the probablity
    pred_prob = model.predict_proba(final_features)
    print(pred_prob)

    if pred:
        msg = f" The employee will leave the organisation , Probablity : {pred_prob[0][1]}"
    else:
        msg = f" The employee will stay with the organisation , Probablity : {pred_prob[0][0]}"

    # output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Result : {} \n {}".format(msg, pred_prob))


# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
#     # prediction = 89
#
#     output = prediction[0]
#     return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
