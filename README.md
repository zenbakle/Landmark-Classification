# Image Classification Model for Landmark Recognition

## Overview

This repository contains an image classification model trained to recognize 50 landmarks with an accuracy of 0.794. The model was developed using transfer learning with the Xception architecture and is deployed as an AWS Lambda function.

## Landmark Data
    The landmark images are a subset of the Google Landmarks Dataset v2.
    It can be found [here](https://drive.google.com/drive/folders/1IrTWdizbkVgPAaEP9HnYK-bnODecOxnS?usp=sharing)

## Building the Docker Image

1. **Prerequisites:**
   - Docker installed and running on your system

2. **Build the image:**
   ```bash
   docker build -t landmark-classifier:latest .
   ```

## Testing the API Gateway

1. **Send a test request:**
   - Use a tool like Postman or curl to send an HTTP request with an image to the API Gateway endpoint.
   - The request body should contain the image url, an example is in the test.py file

2. **Review the response:**
   - The API should return a JSON response with the predicted landmark and its confidence score.


