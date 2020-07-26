import pickle

import env
import flask
import pandas as pd
from flask import render_template, request

app = flask.Flask(__name__)
app.config['DEBUG'] = True


@app.route('/', methods=['GET'])
@app.route('/home', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_resource = request.form['input_resource']
    input_mgr_id = request.form['input_mgr_id']
    input_role_rollup_1 = request.form['input_role_rollup_1']
    input_role_rollup_2 = request.form['input_role_rollup_2']
    input_role_deptname = request.form['input_role_deptname']
    input_role_title = request.form['input_role_title']
    input_role_family_desc = request.form['input_role_family_desc']
    input_role_family = request.form['input_role_family']
    input_role_code = request.form['input_role_code']

    if not input_resource:
        return 'please enter resource'

    if not input_mgr_id:
        return 'please enter manager id'

    if not input_role_rollup_1:
        return 'please enter rollup 1'

    if not input_role_rollup_2:
        return 'please enter rollup 2'

    if not input_role_deptname:
        return 'please enter deptname'

    if not input_role_title:
        return 'please enter title'

    if not input_role_family_desc:
        return 'please enter family desc'

    if not input_role_family:
        return 'please enter family'

    if not input_role_code:
        return 'please enter code'
    # Prepare data
    test = pd.DataFrame(
        [[input_resource, input_mgr_id, input_role_rollup_1, input_role_rollup_2, input_role_deptname,
          input_role_title,
          input_role_family_desc, input_role_family, input_role_code]],
        columns=['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_DEPTNAME', 'ROLE_TITLE',
                 'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE'])

    print(test)

    model = pickle.load(open('../models/catboost_model.pkl', 'rb'))
    model_prediction = model.predict(test)

    print(model_prediction, type(model_prediction), model_prediction[0])

    prediction = 'Access not granted'
    if model_prediction[0] == 1:
        prediction = 'Access granted'

    return render_template('prediction.html', prediction=prediction)


app.run(port=env.port)
