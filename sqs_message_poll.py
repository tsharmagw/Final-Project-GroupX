
#will check for messages in SQS, if message found will run a script
import boto3
import time
import subprocess
import os

def get_message_from_queue(Queue_url,sqs):
	response = sqs.receive_message(
	    QueueUrl=Queue_url,
	    AttributeNames=[
	        'Title'
	    ],
	    MaxNumberOfMessages=1,
	    MessageAttributeNames=[
	        'All'
	    ],
	    VisibilityTimeout=0,
	    WaitTimeSeconds=0
	)
	message = response['Messages'][0]
	print(message)
	data = (message['MessageAttributes']['s3_key']['StringValue'],message['MessageAttributes']['bucketName']['StringValue'])
	#writing data to a temp file
	f = open("/tmp/message_data.txt", "w")
	print(data[0],data[1])
	f.write(data[0]+";"+data[1])
	f.close() 

	receipt_handle=message['ReceiptHandle']
	return(receipt_handle)

if __name__=='__main__':
	success=True
	sqs = boto3.client('sqs',aws_access_key_id='AKIAIAQS3KGWKQL4JJ7A',aws_secret_access_key='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',region_name="us-east-1")
	queue_name="ml2_queue"
	while(success):
		try:
			Queue_url = sqs.get_queue_url(QueueName=queue_name)['QueueUrl']
			handle=get_message_from_queue(Queue_url,sqs)
			subprocess.call(['/home/script.sh']) 
			response = sqs.delete_message(QueueUrl=Queue_url,ReceiptHandle=handle)		
		except Exception as e:
			time.sleep(1)
			print("No message",e)
