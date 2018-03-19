# image_captioning_keras

A simple image captioning demo

## Introduction

Image captioning is an interest subject, it requires both methods from computer vision to understand the content of the image
and a language model from the field of natural language processing to turn the understanding of the image into words 
in the right order. Deep learning methods have achieved state-of-the-art results on examples of this problem.

## Methodology

1. Prepare data and calculate image features using a pre-trained vgg16.
2. Build a LSTM model with image features as input.
3. Build a data generator to generate data for training the model.
4. Train model and test

## Result

The results are not so satisfying. Maybe more training time is needed and a better model with attention. 


## References:
https://machinelearningmastery.com/develop-a-caption-generation-model-in-keras/ </br>
https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
