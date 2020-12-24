// Copyright 2017 Amazon Web Services, Inc. or its affiliates. All rights reserved.

using System;
using Amazon.S3;
using Amazon.S3.Model;

namespace Lab1
{
    // The ReadySetGo class lists the number of buckets in your account.
    public static class ReadySetGo
    {

        public static void Main()
        {
            Console.WriteLine("============================================");
            Console.WriteLine("Welcome to the AWS .NET SDK! Ready, Set, Go!");
            Console.WriteLine("============================================");

            // The Amazon S3 client allows you to manage buckets and objects programmatically.
            IAmazonS3 s3Client = new AmazonS3Client();

            
            try
            {
                ListBucketsResponse response = s3Client.ListBuckets();
                int numBuckets = 0;
                if (response.Buckets != null && response.Buckets.Count > 0)
                {
                    numBuckets = response.Buckets.Count;
                }
                Console.WriteLine("You have " + numBuckets + " Amazon S3 bucket(s)");
            }
            catch (Amazon.S3.AmazonS3Exception S3Exception)
            {
                // AmazonServiceException represents an error response from an AWS service.
                // AWS service received the request but either found it invalid or
                // encountered an error trying to execute it.
                if (S3Exception.ErrorCode != null && (S3Exception.ErrorCode.Equals("InvalidAccessKeyId") || S3Exception.ErrorCode.Equals("InvalidSecurity")))
                {
                    Console.WriteLine("Please check the provided AWS Credentials.");
                    Console.Write("If you haven't signed up for Amazon S3, please visit http://aws.amazon.com/s3");
                    Console.WriteLine(S3Exception.Message, S3Exception.InnerException);
                }
                else
                {
                    Console.WriteLine("Error Message:    " + S3Exception.Message);
                    Console.WriteLine("HTTP Status Code: " + S3Exception.StatusCode);
                    Console.WriteLine("AWS Error Code:   " + S3Exception.ErrorCode);
                    Console.WriteLine("Request ID:       " + S3Exception.RequestId);
                }
            }
            finally
            {
                s3Client.Dispose();
            };
            

        }
    }
}
