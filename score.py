import json
import numpy as np
import pandas as pd
import os
from sklearn.externals import joblib
import azureml.train.automl
import time

def init():
    global model
    # Print statement for appinsights custom traces:
    print ("model initialized" + time.strftime("%H:%M:%S"))
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'outputs/model.pkl')
    model = joblib.load(model_path)

def run(request):
    try:
        request = json.loads(request)
        df = pd.DataFrame([request['data']])
        result = model.predict(df)
        # Log the input and output data to appinsights
        info = {
            "input": request,
            "output": result.tolist()
            }
        print(json.dumps(info))
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        # Log the error to appinsights
        print (error + time.strftime("%H:%M:%S"))
        return error