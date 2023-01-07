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

response = s3.list_objects_v2(Bucket = "hfv.bucket")

lst = []

for item in response["Contents"]:
    lst.append(item['Key'])


N = len(lst)

for j in range(N):

	s3.download_file('hfv.bucket',lst[j],'./Outputs/' + lst[j])
	s3.delete_object(Bucket = 'hfv.bucket', Key = lst[j])


