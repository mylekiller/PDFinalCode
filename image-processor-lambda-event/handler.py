import requests
import boto3
import os

def process(event, context):
	# Load in all the AWS Resources we will be using
	s3 = boto3.resource('s3')
	db = boto3.resource('dynamodb')

	for record in event['Records']:
		packname = record['dynamodb']['Keys']['packName']['S']
		dogname = record['dynamodb']['Keys']['dogName']['S']
		if record['eventName'] == 'MODIFY':
			oldPictureList = record['dynamodb']['OldImage']['picture']['L']
		newPictureList = record['dynamodb']['NewImage']['picture']['L']

		if record['eventName'] == 'MODIFY':
			oldsize = len(oldPictureList)
		else:
			oldsize = 0

		newsize = len(newPictureList)


		for dog in newPictureList[oldsize:newsize]:

			dog = dog['S']
			# Retrieve the picture of interest from the S3 Bucket
			r = requests.get("http://ec2-18-221-201-105.us-east-2.compute.amazonaws.com/process/"+packname+'/'+dogname+'/'+dog)