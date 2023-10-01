from flask import Flask, render_template, request
import pandas as pd
import pickle

main = Flask(__name__)
data = pd.read_csv('Mumbai House Prices.csv')
pipe = pickle.load(open("ridge_model.pkl ", 'rb'))


@main.route('/')
def index():
    region = sorted(data['region'].unique())
    locality = sorted(data['locality'].unique())
    return render_template('index.html', region=region, locality=locality)


@main.route('/predict', methods=['POST'])
def predict():
    region = request.form.get('Region')
    bhk = float(request.form.get('bhk'))
    area = request.form.get('area')
    locality = request.form.get('locality')
    typ = request.form.get('typ')

    print(region, bhk, area, locality, typ)
    inp = pd.DataFrame([[region, bhk, area, locality, typ]], columns=['region', 'bhk', 'area', 'locality', 'type'])
    prediction = pipe.predict(inp)[0]

    return str(prediction)


if __name__ == "__main__":
    main.run(debug=True)