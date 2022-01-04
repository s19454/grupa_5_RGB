from flask import Flask, render_template, request
from flask_api import status
import pickle

app = Flask(__name__)
app.config['SECRET_KEY'] = 'FooBar'

forest_model = pickle.load(open('ml/model.sv', 'rb'))

def predict(data):
    #global forest_model

    res = forest_model.predict(data)

    return res[0]


@app.route('/', methods=['GET'])
def index():

    return render_template('index.html')

@app.route('/query', methods=['GET'])
def query():

    a1 = request.args.get('r')
    a2 = request.args.get('g')
    a3 = request.args.get('b')

    if a1 and a2 and a3:
        data = [[
            a1,
            a2,
            a3
        ]]

        res = predict(data)
        return f'{res}'
    
    else:
        return "All args required!", status.HTTP_400_BAD_REQUEST

@app.route('/json', methods=['POST'])
def json():

    '''
    {
        "r": 1,
        "g": 1,
        "b": 50,
    }
    '''

    json_data = request.json

    a1 = json_data['r']
    a2 = json_data['g']
    a3 = json_data['b']

    data = [[
        a1,
        a2,
        a3
    ]]

    res = predict(data)
    return f'{res}'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8098', debug=True)