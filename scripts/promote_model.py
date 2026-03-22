import os
import mlflow

def promote_model():
    # Set up mlflow tracking URI
    mlflow.set_tracking_uri('http://ec2-13-48-49-1.eu-north-1.compute.amazonaws.com:5000/')

    client = mlflow.MlflowClient()

    model_name = 'yt_chrome_plugin_model'

    # Get the latest version in staging 
    latest_version_staging = client.get_latest_versions(model_name , stages=['Staging'])[0].version

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name , stages=['Production'])
    for version in prod_versions:
        client.transition_model_version_stage(
            name = model_name,
            version=version.version,
            stage='Archived'
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage='Production'

    )
    print(f'Model version {latest_version_staging} promoted to production')

if __name__ == '__main__':
    promote_model()
