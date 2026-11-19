import pickle
import pandas as pd
import numpy as np
import os
import argparse


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

parser = argparse.ArgumentParser()

parser.add_argument('--year', type=int, required=True)
parser.add_argument('--month', type=int, required=True)
args = parser.parse_args()
year = args.year
month = args.month
output_file = 'output.parquet'


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet')


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

y_pred

print(f"Standard deviation for {year} and {month}:", np.std(y_pred))

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_result = pd.DataFrame({
    'ride_id': df['ride_id'],
    'predicted_duration': y_pred
})

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

# Get file size in MB
size_mb = os.path.getsize(output_file) / (1024 * 1024)

print(f"Output file size for {year} and {month}: {size_mb:.2f} MB")

#mean prediction duration
mean_duration = np.mean(y_pred)

print(f"Mean predicted duration for {year} and {month}:", mean_duration)







