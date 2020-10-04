from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error, confusion_matrix
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Dataset

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument("--input-data", type=str)

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Num estimators:", np.int(args.n_estimators))
    run.log("Max depth:", np.int(args.max_depth))

    ws = run.experiment.workspace
    # get the input dataset by ID
    dataset = Dataset.get_by_id(ws, id=args.input_data)
    # load the TabularDataset to pandas DataFrame
    df = dataset.to_pandas_dataframe()

    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("accuracy", np.float(accuracy))
    value = {
       "schema_type": "confusion_matrix",
       "schema_version": "v1",
       "data": {
           "class_labels": ["0", "1"],
           "matrix": confusion_matrix(y_test, model.predict(x_test)).tolist()
       }
    }
    run.log_confusion_matrix(name='Confusion Matrix', value=value)
    os.makedirs('outputs', exist_ok=True)
    # note file saved in the outputs folder is automatically uploaded into experiment record
    joblib.dump(value=model, filename='outputs/model.pkl')

if __name__ == '__main__':
    main()