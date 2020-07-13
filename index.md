# Sentiment Analysis in Financial News
#### ML Modeling with Multinomial Logistic Regression, CNN, LSTM

### Introduction

Sentiment analysis is an application of Natural Language Processing in order to quantify subjective information; models extract information from opinion-based statements and determine the sentiment, or emotion, related to the statement. In particular, models usually identify and label positive, negative, and neutral sentiment from statements and documents.

<p align="center"><img src="./visualizations/sentiment.jpg" alt="Sentiment Analysis"/></p>
<p align="center">Source: <a href="https://www.kdnuggets.com/2018/03/5-things-sentiment-analysis-classification.html">Symeon Symeonidis</a></p>

As with other news headlines, financial news headlines have the same sentiment as that of the information within the news itself. Furthermore, financial news headlines usually closely correlate with investor confidence. Thus, identifying the sentiments of these news headlines can aid predictions on market volatility, trading patterns, and stock prices. Improving the accuracy of models that conduct sentiment analysis on financial news headlines would have many further applications.

### Purpose
The purpose of this project is to build a model that will be able to accurately determine positive, neutral, or negative sentiment in financial news headlines. Our group applies a supervised learning model through multinomial logistic regression in order to achieve the goal. Furthermore, our group applies deep learning models, including convolutional neural networks (CNN) and long short term memory networks (LSTM), to conduct financial sentiment.

As an addendum, we tested the state of the art financial language model on our dataset to see the results and accuracy of a pre-trained model.

### Choosing our Models
We tested a conventional supervised learning model to see what the best accuracy would be for conducting sentiment analysis with models that were not specifically deep-learning models. We chose multinomial logistic regression due to the fewer weights needed to train for the model.

The remaining models were deep learning models. Convolutional neural networks are models that use convolutional layers, which slide a kernel onto the input data. CNNs have had applications in NLP, as word embeddings provide the possibility to use convolution layers to capture semantic information and relations between individual words. Through these convolution layers, partial context can be captured. Thus, we used CNNs as the partial context that could be captured likely would outperform the conventional supervised learning model we tested.

<p align="center"><img src="./visualizations/CNN.jpeg" alt="Convolutional Neural Network"/></p>
<p align="center">Source: <a href="https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53">Sumit Saha</a></p>

We also tested an LSTM model, which is a type of recurrent neural network (RNN). Recurrent neural networks are a type of neural network in which information is fed into the network sequentially. For text and NLP, the RNN creates a hidden state based on running the model on a word, and a new hidden state is generated through using the the previous hidden state and the new word to run the model on. LSTM models are a subtype of RNN models which improve upon the RNN by selectively choosing which information from previous states to remember and which information to forget. We used an LSTM model because LSTM models and RNNs are able to capture sequential information, and, under the assumption that text occurs sequentially, these models may best capture previous context information in order to correctly determine sentiment. 

<p align="center"><img src="./visualizations/RNN.png" alt="Recurrent Neural Network"/></p>
<p align="center">Source: <a href="https://medium.com/deeplearningbrasilia/deep-learning-recurrent-neural-networks-f9482a24d010">Pedro Torres Perez</a></p>

### Dataset
For the financial news dataset, we used the following dataset from kaggle, https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news. The dataset contains 4846 individual datapoints, each one divided between one of three labels - positive, negative, neutral - and the text of the dataset. Note that the length of the text is considerably shorter than that of other sentiment analysis datasets, which may have an effect with the results of our models.

Below are the first three entries of the dataset:
- `According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing . `
- `Technopolis plans to develop in stages an area of no less than 100,000 square meters in order to host companies working in computer technologies and telecommunications , the statement said . ` 
- `The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported .`

The graph below indicates the distribution of labels. Due to the unbalanced nature of the data, it is possible that the results of every model could be affected.

<p align="center"><img src="./visualizations/LabelPercentages.PNG" alt="Label Distribution"/></p>
<p align="center">Distribution of labels</p>

## Multinomial Logistic Regression

### Pre-processing

To preprocess our data, we first took care of stemming--removing word endings resulted from different tenses and word plurality.  Then we removed common stopwords from our news data set. Then we used a count vectorizer to vectorize the words in our dataset by their counts: Each news headline will be a row and each word in our vocabulary will be a column; the vocabulary is the set of words left after stemming and stopwords removal.

<div align="center"><img src="./visualizations/VectorizeExample.PNG" alt="Example Vectorization"/></div>
<div align="center"> Each row is a headline, each column is a word in our vocabulary, and each entry represent the count of the word in the news headline. </div>

#### Model

To create our model, we used 80 percent of our data to train on the logistic regression function provided by sklearn. We ran a 5-fold cross validation algorithm on our model trained with several different regularization level. We were able to achieve the best accuracy results of 70 percent with an inverse regularization value of 0.26.

<div align="center"><img src="./visualizations/accuracy_vs_regularization.png" alt="accuracy_vs_regularization"/></div>
<div align="center"> Showing how mean cross validation accuracy changes as inverse regularization in logistic regression changes </div>

### Results

### Discussion

## Convolutional Neural Network

### Pre-processing

After pre-processing, the 

### Model
For our model, we used the Keras library with TensorFlow backend. As with our other models, it was written in a Jupyter notebook in Google Colab, which are all available on the github repository.

Due to the maximum length of the word count being only 50, our model consisted of three 1-D convolutional layers with the ReLU activation function and a kernel size of 5. In between the convolutional layers, there are maxpool layers with a pool size of 2. Finally, the data is passed as input to a flatten layer and a fully-connected layer that is 128 dimensions long, before finally being passed to the output layer which 

<p align="center"><img src="./visualizations/CNNmodel.png" alt="CNN Model"/></p>

### Results
<p align="center"><img src="./visualizations/CNNloss.png" alt="CNN Loss"/></p>
<p align="center"><img src="./visualizations/CNNacc.png" alt="CNN Accuracy"/></p>
Test Accuracy: 69.278353

### Discussion

## Long Short Term Memory

### Pre-processing

### Model
<p align="center"><img src="./visualizations/LSTMmodel.png" alt="LSTM Model"/></p>

### Results
<p align="center"><img src="./visualizations/LSTMloss.png" alt="LSTM Loss"/></p>
<p align="center"><img src="./visualizations/LSTMacc.png" alt="LSTM Accuracy"/></p>
Test Accuracy: 73.814434

### Discussion

## Comparing our models

## Addendum: State-of-the-art FinBERT model
The state of the art NLP model 

### Model

### Results

### Discussion

## References
Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology, 65(4), 782-796.


