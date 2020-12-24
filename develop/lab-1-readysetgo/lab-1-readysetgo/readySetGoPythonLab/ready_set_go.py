# Copyright 2017 Amazon Web Services, Inc. or its affiliates. All rights
# reserved.

import boto3
from botocore.exceptions import NoCredentialsError,ClientError
import sys

# Before running the code, check that the ~/.aws/credentials file contains your credentials.

def get_bucket_count():
    print("============================================");
    print("Welcome to the AWS Boto3 SDK! Ready, Set, Go!");
    print("============================================");

    try:
        s3 = boto3.resource('s3')
    except NoCredentialsError:
        print("@InvalidCredentials")
        sys.exit()

    try:
        no_of_buckets = len(list(s3.buckets.all()))
        print("You have", str(no_of_buckets), "Amazon S3 buckets.")
        return no_of_buckets

    except ClientError as ex:
        print(ex)
        return 0


if __name__ == '__main__':
    get_bucket_count()
