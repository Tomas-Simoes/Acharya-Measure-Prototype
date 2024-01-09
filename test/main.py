import json


def init(event, context):
    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "http://localhost:8080",
            "Access-Control-Allow-Credentials": True,
            "Access-Control-Allow-Headers": "Origin, Content-Type, Credentials",
            "Access-Control-Allow-Methods": "POST, GET",
            'Content-Type': "application/json"
        },
        "body": json.dumps({"message": "hello world"})
    }
