import json
import boto3
def put_message_in_queue(sqs,key_name,bucketName,QueueName='ml2_queue'):
        Queue_url = sqs.get_queue_url(QueueName=QueueName)['QueueUrl']
        response = sqs.send_message(
                    QueueUrl=Queue_url,
                    DelaySeconds=10,
                    MessageAttributes={
                        'Title': {
                            'DataType': 'String',
                            'StringValue': 'ml2_pipeline'
                        },
                        's3_key': {
                                'DataType':'String',
                            'StringValue': key_name,

                       },
                        'bucketName':{
                                'DataType':'String',
                                'StringValue':bucketName

                        },
                    },
                    MessageBody=(
                        'Queue_Data'
                    )
                )
def lambda_handler(event, context):
    sqs = boto3.client('sqs')
    bucketName=event['Records'][0]['s3']['bucket']['name']
    key_name= event['Records'][0]['s3']['object']['key']
    put_message_in_queue(sqs,key_name,bucketName)
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }

