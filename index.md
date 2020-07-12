# Sentiment Analysis in Financial News
## ML Modeling with Multinomial Logistic Regression, CNN, LSTM

### Introduction

Sentiment analysis is an application of Natural Language Processing in order to quantify subjective information; models extract information from opinion-based statements and determine the sentiment, or emotion, related to the statement. In particular, models usually identify and label positive, negative, and neutral sentiment from statements and documents.

As with other news headlines, financial news headlines have the same sentiment as that of the information within the news itself. Furthermore, financial news headlines usually closely correlate with investor confidence. Thus, identifying the sentiments of these news headlines can aid predictions on market volatility, trading patterns, and stock prices. Improving the accuracy of models that conduct sentiment analysis on financial news headlines would have many further applications.

### Purpose
The purpose of this project is to build a model that will be able to accurately determine positive, neutral, or negative sentiment in financial news headlines. Our group applies a supervised learning model through multinomial logistic regression in order to achieve the goal. Furthermore, our group applies deep learning models, including convolutional neural networks (CNN) and long short term memory networks (LSTM), to conduct financial sentiment.

As an addendum, we tested the state of the art financial language model on our dataset to see the results and accuracy of a pre-trained model.

### Choosing our Models
We tested a conventional supervised learning model to see what the best accuracy would be for conducting sentiment analysis with models that were not specifically deep-learning models. We chose multinomial logistic regression due to the fewer weights needed to train for the model.

The remaining models were deep learning models. Convolutional neural networks are models that use convolutional layers, which slide a kernel onto the input data. CNNs have had applications in NLP, as word embeddings provide the possibility to use convolution layers to capture semantic information and relations between individual words. Through these convolution layers, partial context can be captured. Thus, we used CNNs as the partial context that could be captured likely would outperform the conventional supervised learning model we tested.

We also tested an LSTM model, which is a type of recurrent neural network (RNN). Recurrent neural networks are a type of neural network in which information is fed into the network sequentially. For text and NLP, the RNN creates a hidden state based on running the model on a word, and a new hidden state is generated through using the the previous hidden state and the new word to run the model on. LSTM models are a subtype of RNN models which improve upon the RNN by selectively choosing which information from previous states to remember and which information to forget. We used an LSTM model because LSTM models and RNNs are able to capture sequential information, and, under the assumption that text occurs sequentially, these models may best capture previous context information in order to correctly determine sentiment. 

### Dataset

## Multinomial Logistic Regression

### Pre-processing

### Model

### Results

### Discussion

## Convolutional Neural Network

### Pre-processing

### Model

### Results

### Discussion

## Long Short Term Memory

### Pre-processing

### Model

### Results

### Discussion

## Comparing our models

## Addendum: State-of-the-art FinBERT model

### Model

### Results

### Discussion
