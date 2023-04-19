# Project Name
Student Math Score Prediction

## Project Description
This project aims to predict the math score of a student based on various input parameters. The model selection is based on the lowest Mean Squared Error (MSE) value. The input parameters include numerical and categorical features.

## Installation
To install the necessary dependencies, run the following command:
pip install -r requirements.txt

## Pipeline
To train and test the model, a pipeline was created to perform feature engineering and data manipulation to organize the input data. The data input is passed through a transformation using num_pipeline and cat_pipeline.

- num_pipeline: This pipeline performs the following operations:
    - Missing values are imputed using the median strategy.
    - Scaling is done using StandardScaler.

- cat_pipeline: This pipeline performs the following operations:

    - Missing values are imputed using the most frequent strategy.
    - OneHotEncoding is performed.
    - Scaling is done using StandardScaler with_mean=False.

## Usage
To test the project, run the app.py file on your local host. You can use the following endpoints to make predictions and view model performance:

- /predictdata: This endpoint is used to make predictions based on the input parameters.
- /retrain: This endpoint is used to retrain the model based on new data. Here, you can also view the model's performance. If the new model has a lower MSE value, it will replace the old model in the production environment. If not, the old model will be kept in the production environment.
Installation

To install the dependencies for this project, run the following command:

bash
Copy code
pip install -r requirements.txt

## Conclusion
This project provides an easy-to-use and accurate method to predict a student's math score based on various input parameters. The model selection process ensures that the best model is chosen based on the lowest MSE value. With the ability to retrain the model on new data, this project can provide continuous improvement to the model's performance.


## References
https://github.com/krishnaik06/mlproject
https://www.youtube.com/watch?v=1m3CPP-93RI&t=600s


MLOPS: A COMPLETE GUIDE TO MACHINE LEARNING OPERATIONS | MLOPS VS DEVOPS
https://ashutoshtripathi.com/2021/08/18/mlops-a-complete-guide-to-machine-learning-operations-mlops-vs-devops/


