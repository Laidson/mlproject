from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle


from sklearn.preprocessing import StandardScaler

from src.pipeline.predic_pipeline import CustomData, PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.exception import CustomException


application=Flask(__name__)

app = application

##Route for a hoem page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score')),
        )

        pred_df = data.get_data_as_data_frame()
        pred_df.to_csv('pred_df.csv', index=False)
        print(pred_df)

        predict_pipeline = PredictPipeline()
        print(predict_pipeline)

        results = predict_pipeline.predict(pred_df)
        print(results)


        return render_template('home.html',results=results[0])

# Define route for retraining the model
@app.route('/retrain', methods=['GET','POST'])

def retrain_model():
    if request.method == 'GET':
        return render_template('retrain.html', new_mse='', old_mse='', improve='')
    
    elif request.method == 'POST':
        train_pipe = TrainPipeline()
        metric = train_pipe.main()
        improve = (metric['new_metric'] - metric['old_metric'])*1000

        return render_template('retrain.html', new_mse=metric['new_metric'], 
                                                old_mse=metric['old_metric'],
                                                 improve = improve)
    else:
        return render_template('home.html')

pass


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)

