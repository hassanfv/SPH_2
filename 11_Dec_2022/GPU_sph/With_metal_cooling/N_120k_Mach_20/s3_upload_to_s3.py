import os
import boto3
import pickle
import time
import glob
import numpy as np
from decouple import config  # pip install python-decouple

#-----------------
# Authentication #
#-----------------
AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')
AWS_REGION_NAME = config('AWS_REGION_NAME')

# create an s3 access object
s3 = boto3.client('s3', aws_access_key_id     = AWS_ACCESS_KEY_ID, 
			aws_secret_access_key = AWS_SECRET_ACCESS_KEY, 
			region_name           = AWS_REGION_NAME)


# Uploading the files into the s3 bucket
filz = np.sort(glob.glob('./Outputs/*.csv'))

for i in range(len(filz)):

	nam = filz[i].split('/')[-1]

	s3.upload_file(Filename = filz[i], Bucket = 'hfv.bucket', Key = nam)
	
	os.remove(filz[i])





