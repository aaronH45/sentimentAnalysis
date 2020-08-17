# Sentiment Analysis in Financial News
#### ML Modeling with Multinomial Logistic Regression, CNN, LSTM

### Introduction

Sentiment analysis is an application of Natural Language Processing in order to quantify subjective information; models extract information from opinion-based statements and determine the sentiment, or emotion, related to the statement [7]. In particular, models usually identify and label positive, negative, and neutral sentiment from statements and documents.

<p align="center"><img src="./visualizations/sentiment.jpg" alt="Sentiment Analysis"  width="640"/></p>
<p align="center">Source: <a href="https://www.kdnuggets.com/2018/03/5-things-sentiment-analysis-classification.html">Symeon Symeonidis</a></p>

As with other news headlines, financial news headlines have the same sentiment as that of the information within the news itself. Furthermore, financial news headlines usually closely correlate with investor confidence [2]. Thus, identifying the sentiments of these news headlines can aid predictions on market volatility, trading patterns, and stock prices. Attaining models that can consistently label the correct sentiment of financial news headlines with a high accuracy would have many further applications, including measuring the general investor sentiment about the market and predict expected economic trends [6].

### Purpose
The purpose of this project is to build a model that will be able to accurately determine positive, neutral, or negative sentiment in financial news headlines. Our group applies a supervised learning model through multinomial logistic regression in order to achieve the goal. Furthermore, our group applies deep learning models, including convolutional neural networks (CNN) and long short term memory networks (LSTM), to conduct financial sentiment. We then compare the results of all three models and conclude what the strengths of each model are.

### Choosing our Models
We tested a conventional supervised learning model to see what the best accuracy would be for conducting sentiment analysis with models that were not specifically deep-learning models. We chose multinomial logistic regression due to the fewer weights needed to train for the model, and the documentation it has had with published papers [13].

The remaining models were deep learning models. Convolutional neural networks are models that use convolutional layers, which slide a kernel onto the input data. CNNs have had applications in NLP, as word embeddings provide the possibility to use convolution layers to capture semantic information and relations between individual words [7]. Through these convolution layers, partial context can be captured. Thus, we used CNNs as the partial context that could be captured likely would outperform the conventional supervised learning model we tested [7].

<p align="center"><img src="./visualizations/CNN.jpeg" alt="Convolutional Neural Network"  width="640"/></p>
<p align="center">Source: <a href="https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53">Sumit Saha</a></p>

We also tested an LSTM model, which is a type of recurrent neural network (RNN). Recurrent neural networks are a type of neural network in which information is fed into the network sequentially. For text and NLP, the RNN creates a hidden state based on running the model on a word, and a new hidden state is generated through using the the previous hidden state and the new word to run the model on [11]. LSTM models are a subtype of RNN models which improve upon the RNN by selectively choosing which information from previous states to remember and which information to forget. We used an LSTM model because LSTM models and RNNs are able to capture sequential information, and, under the assumption that text occurs sequentially, these models may best capture previous context information in order to correctly determine sentiment [4]. 

<p align="center"><img src="./visualizations/RNN.png" alt="Recurrent Neural Network"  width="640"/></p>
<p align="center">Source: <a href="https://medium.com/deeplearningbrasilia/deep-learning-recurrent-neural-networks-f9482a24d010">Pedro Torres Perez</a></p>

### Dataset
For the financial news dataset, we used the Financial Phrasebank dataset, which can be found in a refined form on kaggle, https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news. The dataset contains 4846 individual datapoints, each one divided between one of three labels - positive, negative, neutral - and the text of the dataset. Note that the length of the text is considerably shorter than that of other sentiment analysis datasets, which may have an effect with the results of our models.

Below are the first three entries of the dataset:
- `According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing . `
- `Technopolis plans to develop in stages an area of no less than 100,000 square meters in order to host companies working in computer technologies and telecommunications , the statement said . ` 
- `The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported .`

<p align="center"><img src="./visualizations/wordcloud.png" alt="Word cloud of the stemmed dataset"/></p>
<p align="center">Word cloud of the stemmed dataset</p>

The graph below indicates the distribution of labels. Due to the unbalanced nature of the data, it is possible that the results of every model could be affected.

<p align="center"><img src="./visualizations/LabelPercentages.PNG" alt="Label Distribution"/></p>
<p align="center">Distribution of labels</p>

## Multinomial Logistic Regression

### Pre-processing
To preprocess our data, we tokenized and stemmed each headline. We treated each word in the dataset as a token, and we used the common Porter Stemmer to stem the words, ignoring any inflections that the word may have. Thus, the headlines are converted into lists of root words, with tenses and plurality removed for consistency [10]. Afterwards, we removed common stopwords from our data using a preset list of English stopwords. We applied the common bag-of-words model and used a count vectorizer to vectorize the words in our training dataset by the number of times each word appeared. We used a minimum frequency of 5, so the model ignored words that appeared less than 5 times throughout all the training headlines. In addition to word counts, we processed the headlines using a term frequency- inverse document frequency (TFIDF) matrix. This process assigns each word in a headline a frequency proportional to number of times it appears in a headline and the number of headlines the word appears in. TFIDF reflects the importance of a word to the document and the corpus of the dataset as the inverse relationship allows for common words present in all documents to not have much information [9]. Comparatively, a word with high frequency in one headline but relatively low frequency in the other headlines would provide more information regarding the sentiment of that headline.

Below are some visualizations of the pre-processing steps we took for our logistic regression model.

#### Porter Stemmer and Count Vectorizer

|Before|Porter Stemmer|After|Count Vectorizer|Key|
|----------|:--------:|------|:------:|------|
|According|⭢|accord|⭢|784|
|to|⭢|to|⭢|N/A|
|Gran|⭢|gran|⭢|3211|
|the|⭢|the|⭢|N/A|
|company|⭢|compani|⭢|1826|

<div align="center"> Example words being stemmed and stop words being removed </div>

#### TF-IDF Matrix

|          | accord | area | compani | comput | develop | gran | grow |
|---|---|---|---|---|---|---|---|
|Sentence 1|  0.342369 | 0 | **0.48719673** | 0 | 0 | 0.342369 | 0.342369 | 
|Sentence 2|  0 | 0.24244659 | **0.17250275** | 0.24244659 | 0.24244659 | 0 | 0 |

<div align="center"> TF-IDF matrix weighting the first few words of the corpus in our dataset</div>
<div align="center"> (Company has the has uniqueness score for the first sentence because it is used twice, but lowest for the second sentence because it has also been used twice in the first sentence compared to the other words which only appear in the second sentence)</div>

#### Model
To create our model, we used the logistic regression function provided by the sklearn library [13]. Our code was conducted on Google Colab, and is available on the github repository. We used 80 percent of our data to train on the model, and we used the other 20 percent to test the accuracy of our model. We ran a 5-fold cross validation algorithm on our model trained with several different regularization level. We were able to achieve the best accuracy results of 70 percent with an inverse regularization value of 0.26. 

<div align="center"><img src="./visualizations/accuracy_vs_regularization.png" alt="accuracy_vs_regularization"/></div>
<div align="center"> Showing how mean cross validation accuracy changes as inverse regularization in logistic regression changes </div>

### Results
We compared the number of correctly classified financial news headlines with the total number of the test set. Our model predicted **37%** of the test headlines correctly. When the headlines were processed with TFIDF, we saw a slight decrease in the accuracy to 36%. Below we charted the 25 most positivley and negativley influential words for each sentiment. 

<p align="center"><b>Most influential neutral words</b></p>
<div align="center"><img src="./visualizations/neutral.png" alt="neutral words" width="640"/></div>
<p align="center"><b>Most influential negative words</b></p>
<div align="center"><img src="./visualizations/negative.png" alt="negative words" width="640"/></div>
<p align="center"><b>Most influential positive words</b></p>
<div align="center"><img src="./visualizations/positive.png" alt="positive words" width="640"/></div>

### Discussion
We believe the relatively low accuracy of the logistic regression model can be attributed to the relatively low sample size and the majority of the dataset being classified as neutral. In particular, the logistic regression model may have suffered as the training set produced a vocab of only 2340 words, meaning many of the words that appeared in the test set headlines may have been unknown to the model and therefore, was lost information. The TFIDF pre-processing may have had a negligible effect due to the nature of the financial text corpus; in general, financial text may have different frequencies for the occurrences of different words and thus would not have much impact on the dataset. Finally, the bag-of-words model likely captured insufficient information regarding the relationships and semantic meaning of text, leading to a low overall accuracy [5].

## Convolutional Neural Network

### Pre-processing
For pre-processing the text of the dataset, we first removed the punctuation from the text in the dataset before lowercasing each token. Then, we removed the common English list of stopwords from our dataset from the nltk.corpus library. 

For the labels, each label of each data point is treated as a one-hot three-dimensional vector embedding, depending on the value of the label.

#### Word2vec
Afterwards, we used the common word to vector embedding, word2vec. Word2vec is a type of word-vector embedding developed by Google, which is trained on a dataset of Google News. The general word2vec algorithm takes a word as input and outuputs the probability of every word in the corpus to appear in the input word's surroundings. As a result, the algorithm allows for words in similar contexts to have word vectors with high cosine similarity, as the word vectors will have similar probabilities. As a result, word2vec is able to capture semantic meaning of words, and serve as a viable input to our CNN model [5].

<p align="center"><img src="./visualizations/word2vec.png" alt="skip-gram word2vec"/></p>
<p align="center">Word2vec algorithm</p>
<p align="center">Source: <a href="https://petamind.com/word2vec-with-gensim-a-simple-word-embedding-example/">petamind</a></p>

We used the pre-trained word2vec model to pre-process our dataset. The word to vector embedding maps every word to a 300-dimensional vector. The maximum length of a sequence in our dataset is 47, so we set the maximum length of our input to be 50. For each word in a datapoint, if a mapping in word2vec existed, we mapped it and appended it to our input; if such a mapping did not exist, we randomized the entries in our input. Finally, each datapoint was padded with 0's to ensure that the maximum sequence length is 50. Thus, the size of the input for every datapoint is 50x300.

After pre-processing, the dataset becomes an input of Nx50x300, in which N is the number of datapoints of our dataset. We split the dataset into 80% training, 10% validation, 10% testing.

### Model
For our model, we used the Keras library with TensorFlow backend. As with our other models, it was written in a Jupyter notebook in Google Colab, which are all available on the github repository.

Our CNN model took inspiration from conventional CNN models used in other fields, such as computer vision as well as models from documented papers [7]. However, due to the maximum length of the word count being only 50, our model consisted of three 1-D convolutional layers rather than four 1-D convolutional layers. Our model uses 1-D convolutional layers because we believe that the 1-D layers would capture more information from words in a sequence. On the other hand, 2-D convolutional layers would likely capture information from the same word, reducing the importance of information between the words. These layers have a the ReLU activation function and a kernel size of 5. The convolutions span through the entire sequence in order to capture as much contextual information as possible. In between the convolutional layers, there are maxpool layers with a pool size of 2. Finally, the data is passed as input to a flatten layer and a fully-connected layer that is 128 dimensions long, before finally being passed to the output layer which uses the softmax function to produce 3 softmaxed outputs, which represent the 3 labels of the dataset. Categorical cross entropy is used as our loss function. There are a total of 264,419 trainable parameters.

In order to improve the accuracy of the model, we altered the kernel size, maximum pool size, as well as added and removed CNN layers. We also implemented the Adam optimizer with a learning rate of 0.00025 as it converged faster and had better results than stochastic gradient descent. The model below proved to be the most effective in achieving consistently high accuracies.

<p align="center"><img src="./visualizations/CNNmodel.png" alt="CNN Model"/></p>
<p align="center">Our CNN model</p>

To reduce overfitting, dropout, regularization, and early stopping were added to the model. Dropout with a factor of 0.3 was added before the convolutional layers as it reduced overfitting the most, and another dropout layer was added between the fully-connected layers at the end of the model. L2 and bias regularizers were added to the fully-connected layers, and the accuracy of the model was highest with a factor of 0.04. Due to the short length of each data point, early stopping occured once the validation loss did not decrease in 3 epochs.

### Results
Our evaluation metric was the accuracy of our sentiment prediction. For the test set, we compared the number of correctly classified financial news headlines with the total number of the test set. Though we did not initially set a target accuracy to achieve, after seeing the accuracy of our multinomial logistic regression model, we hoped to achieve an accuracy over 66%. In the end, we were able to achieve an accuracy of **70.103091%**.

Below are visualizations that plot the loss and the accuracy versus epochs during the training of the model.
<p align="center"><img src="./visualizations/CNNloss.png" alt="CNN Loss"/></p>
<p align="center"><img src="./visualizations/CNNacc.png" alt="CNN Accuracy"/></p>

### Discussion
We speculate that there are several reasons why we achieved such an accuracy and how we could improve our model. First of all, as mentioned in the discussion of the dataset, the imbalance of the number of labels could skew our results and model [7]. Furthermore, the financial corpus and dataset is different than that of the general corpus of text. Thus, since we used the pre-trained Google word2vec vector embedding, we likely conducted an incorrect mapping of the words to a vector space which was not sufficient for capturing semantic relationships between words in the financial dataset. To improve our model, a word embedding with a model that is pre-trained on the financial dataset, such as FinBERT word embeddings, could likely improve our results [3].

## Long Short Term Memory

### Pre-processing
The pre-processing needed for the LSTM model was much less compared to the other two models. We used the text to sequences method of the keras library to pre-process the dataset. The texts to sequences method converts the text into a sequence of integers, and only the words that are known by the tokenizer is taken into account. 

For the labels, each label of each data point is treated as a one-hot three-dimensional vector embedding, depending on the value of the label.

### Model
As mentioned earlier, the LSTM model is a type of Recurrent Neural Network, in which each hidden state is taken as additional input to calculate the next hidden state. The LSTM model improves upon the RNN because, as hidden states continue, RNNs tend to forget the information gained from hidden states that occured very early on in the sequence; this is known as catastrophic forgetting. Thus, the LSTM improves upon the issue by selectively choosing which information to keep and which information to forget [8].

Our model was written in a jupyter notebook, and is available on the github repository. Our model consists of an embedding layer to convert the text input into an embedding, two LSTM layers, and a dense, fully-connected layer that outputs the softmaxed outputs of the probability of the three label predictions of each datapoint. The embedding dimension of the input is 128 dimensions, and the LSTM dimension is 196 dimensions long. We use categorical cross entropy as our loss function, and we used the Adam optimizer as it provided better results than that of stochastic gradient descent. Overall, there were 819,503 trainable parameters in the model. To improve our model, we tested adding fully connected layers, as well as adding and removing LSTM layers. In the end, the following model below proved to be the most effective in in achieving high accuracies.

<p align="center"><img src="./visualizations/LSTMmodel.png" alt="LSTM Model"/></p>

In order to prevent overfitting, a spatial dropout layer was added before the LSTM layers [4]. The spatial dropout layer was found to be the best at a factor of 0.4. Furthermore, recurrent dropout and dropout was added to the LSTM layers. Due to the fact that our model had two LSTM layers, the dropout of each LSTM layer was the best at a factor around 0.4, and the recurrent dropout was found to be best around a factor of 0.5. Early stopping was not added to this model since it only required a few epochs to train and generalize well.

### Results
As with our other models, our evaluation metric was the accuracy of our sentiment prediction. We tried to see if we could improve upon the accuracy seen by our CNN model. 
In the end, we were able to achieve a test accuracy of **73.814434%**.
  
Below are the visualizations that plot the loss and the accuracy versus epochs during the training of the model.
<p align="center"><img src="./visualizations/LSTMloss.png" alt="LSTM Loss"/></p>
<p align="center"><img src="./visualizations/LSTMacc.png" alt="LSTM Accuracy"/></p>

### Discussion
The LSTM model consistently had above a 72% accuracy, indicating the effectiveness of Recurrent Neural Networks in improving upon NLP tasks due to its natural recurrent and sequential nature. However, we speculate that the LSTM model may not be extremely effective in this dataset due to the fact that the lengths of the headlines are much short, at a maximum sequence length of only 47. Thus, the LSTM may have a negligible effect on improving the dataset compared to solely using Recurrent Neural Networks. Furthermore, since the LSTM is only trained on sequential text in one direction, relationships in text that could occur in reverse are ignored, which could also impact the accuracy of our models [4].

## Conclusion / Comparing our models
If the labels were perfectly distributed, then randomly guessing the correct sentiment would be a baseline of 33%. However, due to the imbalance of labels in our dataset, it is likely that randomly guessing would provide a different accuracy. Regardless, our multinomial logistic regression model performed just slightly better than randomly guessing in a perfectly distributed dataset, at an accuracy of around 36%. The model did not generalize well, and overall, using a supervised model that was not also a deep-learning model provided the worst results in our dataset. We conclude that simply using a bag-of-words model and implementing the TFIDF pre-processing algorithm is not enough to capture semantic information and contextual relationships that can successfully and consistently determine the correct label for a sequence of text.

<p align="center"><img src="./visualizations/Compare.PNG" alt="Label Distribution"/></p>

The CNN model drastically improved upon the results of the logistic regression, as we were able to achieve above a 70% accuracy for our model. The model converged fast and trained fast, requiring only about 12 epochs on average to train. The model generalized well, and we can reasonably conclude that the convolutions over the word embeddings of our model proved to be able to capture contextual and semantic information that aided in predicting the sentiment of each data point. Though not perfect, the convolutions were able to retain more information than that of our supervised logistic regression model.

Finally, the increase of the LSTM model is not negligible, as we were able to achieve almost a 74% accuracy for our model. Fewer epochs were required to train our LSTM model, but it is worth noting that, due to the temporal nature of RNNs, each epoch took a considerable amount of time longer to train than an individual epoch for the CNN model. The recurrent nature of our LSTM model proved to be more effective in predicting the sentiment of each data point, likely due to the fact that text is sequential and LSTM and RNN models take advantage of the sequential nature of data. In the end, our LSTM model proved to be the best model with the highest accuracy out of all three of our models, and recurrence for NLP tasks, such as sentiment analysis, is very effective.

In the end, our models, especially our deep learning models, proved to have good results that indicate the strength of CNNs and RNNs in sentiment analysis. Based on our results, we can conclude that deep learning models such as the CNN and LSTM are effective in predicting the sentiment analysis of financial text. CNNs can capture relationships between words through convolutional kernels, and RNNs can capture sequential information successfully through its recursive architecture. Future work should focus on improving models to and achieving higher accuracies before anyone is able to apply these models for other applications in the financial field. 

### Addendum: Where to go from here?
Recent advances in sentiment analysis indicate the strength of a new architecture that is able to improve upon sentiment analysis tasks. In the past few years, the transformer, which uses multihead self-attention to capture information about relationships between features in a sequence, has proven to be extremely effective in capturing contextual information of a sequence, much more so than any other architecture. By using self-attention, an internal representation of a sequence is created, which is extremely effective in capturing relationships between words [12].

<p align="center"><img src="./visualizations/attention.png" alt="Self-attention" width="640"/></p>
<p align="center">Self attention of the sequence being able to capture relationships such as pronoun antecedent relationships</p>
<p align="center">Source: <a href="https://arxiv.org/pdf/1706.03762.pdf">Vaswani et al.</a></p>

Thus, transfer learning and pre-training a transformer language model before finetuning the model on downstream tasks, such as sentiment analysis, is the topic of most future work and studies [3]. Moreover, financial sentiment analysis has already seen vast improvements, having an accuracy of 86% through the use of transformer architectures and transfer learning [1].

## References
[1] Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. ArXiv, abs/1908.10063.

[2] Baker, Malcolm, and Jeffrey Wurgler. "Investor sentiment in the stock market." Journal of economic perspectives 21.2 (2007): 129-152.

[3] Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT.

[4] D. Li and J. Qian, "Text sentiment analysis based on long short-term memory," 2016 First IEEE International Conference on Computer Communication and the Internet (ICCCI), Wuhan, 2016, pp. 471-475, doi: 10.1109/CCI.2016.7778967.

[5] Karani, Dhruvil. “Introduction to Word Embedding and Word2Vec.” Medium, Towards Data Science, 2 Sept. 2018, towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa.

[6] Krishnamoorthy, S. (2018). Sentiment analysis of financial news articles using performance indicators. Knowledge & Information Systems, 56(2), 373–394. https://doi.org/10.1007/s10115-017-1134-1

[7] Nedjah, N., Santos, I. & de Macedo Mourelle, L. Sentiment analysis using convolutional neural network via word embeddings. Evol. Intel. (2019). https://doi-org.prx.library.gatech.edu/10.1007/s12065-019-00227-4

[8] Schak M., Gepperth A. (2019) A Study on Catastrophic Forgetting in Deep LSTM Networks. In: Tetko I., Kůrková V., Karpov P., Theis F. (eds) Artificial Neural Networks and Machine Learning – ICANN 2019: Deep Learning. ICANN 2019. Lecture Notes in Computer Science, vol 11728. Springer, Cham

[9] Stecanella, Bruno. “What Is TF-IDF?” MonkeyLearn Blog, 14 July 2020, monkeylearn.com/blog/what-is-tf-idf/.

[10] *Stemming and Lemmatization*, nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html.

[11] “Understanding LSTM Networks.” Understanding LSTM Networks -- Colah's Blog, colah.github.io/posts/2015-08-Understanding-LSTMs/.

[12] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., & Polosukhin, I. (2017). Attention is All you Need. ArXiv, abs/1706.03762.

[13] W. P. Ramadhan, S. T. M. T. Astri Novianty and S. T. M. T. Casi Setianingsih, "Sentiment analysis using multinomial logistic regression," 2017 International Conference on Control, Electronics, Renewable Energy and Communications (ICCREC), Yogyakarta, 2017, pp. 46-49, doi: 10.1109/ICCEREC.2017.8226700.
