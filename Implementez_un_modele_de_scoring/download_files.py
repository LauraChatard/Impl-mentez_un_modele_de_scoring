import boto3

def download_files():
    s3 = boto3.client('s3')
    bucket_name = 'elasticbeanstalk-eu-north-1-182399693743'
    
    files = [
        'model.pkl',
        'imputer.pkl',
        'scaler.pkl',
        'feature_names.json'
    ]
    
    for file_name in files:
        s3.download_file(bucket_name, file_name, file_name)
        print(f"Downloaded {file_name}")

if __name__ == "__main__":
    download_files()
