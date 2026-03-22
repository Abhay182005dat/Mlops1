import mlflow
import pytest
import pandas as pd
import pickle
from mlflow.tracking import MlflowClient

# Set your remote tracking URI
mlflow.set_tracking_uri('http://ec2-13-51-251-184.eu-north-1.compute.amazonaws.com:5000/')

@pytest.mark.parametrize('model_name , stage , vectorizer_path' , [
    ('yt_chrome_plugin_model' , 'staging' , 'tf-df_vectorizer.pkl')
])
def test_model_with_vectorizer(model_name , stage , vectorizer_path):
    client = MlflowClient()

    # Get the latets version in the specified stage 
    latest_version_info = client.get_latest_versions(model_name , stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None

    assert latest_version is not None, f"No model found in the '{stage}' stage for '{model_name}' "

    try:
        # load the latest version of the model
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # load the vectorizer
        with open(vectorizer_path , 'rb') as file:
            vectorizer = pickle.load(file)
        
        # create a dummy input for the model
        input_text = "Hey how are you"
        input_data = vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray() , columns = vectorizer.get_feature_names_out())

        # predict using the model
        prediction = model.predict(input_df)

        # verify the input shape matches the vectorizer's output 
        assert input_df.shape[1] == len(vectorizer.get_feature_names_out()) , "Input feature count mismatch"

        # verify the output shape (assuming buinary classification with a single input)
        assert len(prediction) == input_df.shape[0], "Output row count mismatch"

        print(f"Model '{model_name}' version {latest_version} successfully processed the dummy input text")

    except Exception as e:
        pytest.fail(f'Model test failed with error : {e} ') 