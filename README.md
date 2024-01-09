# Emotions_dataset_ML
## Introduction
The purpose of this project is to create a statistical model that can identify the emotional sentiment of a short sentence as accurately as possible. The possible scope of this project is greatly restrained by the dataset that was chosen, as it only includes 6 emotional sentiments for the algorithm to identify. Emotional sentiment is very nuanced and there are much more than just 6 emotional sentiments that one may express, but the simplicity of this dataset will make it easier to create a basic functional product. This notebook was also an experiment in using copilot to accelerate my workflow and to make my code more accurate.

While this project is being done for learning, there are many practical applications of sentiment classification. This model may be used to identify the sentiment of customers towards a brand or product quickly and efficiently. It would also be useful as a light-weight option to label the sentiment of a large corpus of text samples as a part of a larger machine learning pipeline.

The text samples are divided into 6 categories depending on their emotional sentiment: Anger, Sadness, Love, Joy, Fear, and Surprise. These sentiments are quite general; most other emotions that people express in a typical discourse can be grouped into one of them. For this reason, they are a good representation of a generalized emotional range. The data was retrieved from [Kaggle](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp). The original uploader has already processed and cleaned the data, rendering it into a format that is ready for tokenization and analysis via machine learning. The data has also already been split into training, validation, and test sets of appropriate size, which is potentially both good and bad. It may save effort in the short term, but for processes where we can't plug in a pre-split validation set, we may have to write extra code to recombine them.

## Data Explaination
Most of the samples in the data were labeled as 'joy' or 'sadness'. The least common label is 'surprise'. There is a significant imbalance between the distribution of labels, which may have affected the performance of the final model.
![](https://github.com/Davidkeebler/Emotions_dataset_ML/blob/main/img/labeldistro.png)

Joy and Sadness had the most unique words associated with them. Joy had ~5000 unique words in its samples that were not contained in samples with other labels. Surprise had the fewest unique words, which tracks with the fact that it also has the fewest total samples.
![](https://github.com/Davidkeebler/Emotions_dataset_ML/blob/main/img/numwords.png)

With stopwords and the most common shared words between all samples removed, the following set of charts describes the distribution of the most common words for each label:
![](https://github.com/Davidkeebler/Emotions_dataset_ML/blob/main/img/wordfreq.png)

## Methodology
This project acomplished its goals through the use of a simple neural network with a single hidden layer and a basic RELU activation function. The basic parameters that I started with were taught to me in [this lesson](https://github.com/learn-co-curriculum/dsc-introduction-to-keras-lab/tree/solution) from flatiron school as a basic architecture that works well for text classification. No extra cleaning or splitting of the data was necessary, as this had already been performed by the aggregator on kaggle. The basic tokenizer included in Keras was used to vectorize the text with an inventory of 2000 words. No special tokenization techniques were used, and the resulting tokenizer model was saved to a json file so it can be loaded and used by other applications. 
A parameter gridsearch was attempted with the neural network, but none of the attempted parameter shifts were better than the basic parameters that we started with.
Construction of a Naive Bayes model was also attempted in the notebook, but its performance was not comparable to the neural network and it was rejected.

## Results
![](https://github.com/Davidkeebler/Emotions_dataset_ML/blob/main/img/acc_loss_f1.png)
The final model achieved an accuracy of 88% on the test data with a final F1 score of 0.3. Likely due to the imbalance in labels, the model struggles to properly identify surprise, which was the least common emotion in the dataset. It also has a tendency to occasionally confuse fear and anger, which may be due to their similar inventory of words. The following are charts of accuracy and loss over the course of the model's training, and a confusion matrix that shows its results:
![](https://github.com/Davidkeebler/Emotions_dataset_ML/blob/main/img/confusionmatrix.png)

## Analysis and Conclusions
This model is useful for the analysis of customer sentiment expressed on short form social media platforms such as twitter. It has the limitation of only remaining accurate for samples roughly 150 characters in length, so it is best suited to analyzing the sentiment of short messages. The most interesting use of this model would be to create a dashboard that measures customer feedback in real time in response to the activities of a company.

## Next Steps
There are still many ways the model could be improved. Notably, other notebooks that work with this data achieved an accuracy slightly higher than mine - I managed 88%, and the highest I could find was 92%. Here are a few steps I might take to further improve my process:
- Experiment more with the structure of the neural network. I used a very basic architecture I've seen others use for similar text identification tasks, and it is possible that different activation functions or connection types could improve accuracy.
- One hot encode words that are unique to each emotional sentiment. This is not guaranteed to increase the model's accuracy, but it is the logical next step to try.
- Experiment with different tokenization techniques. This notebook just used the most simple tokenizer available in the Keras library. It is possible that a more complex tokenization technique could preserve more of the information in the text and slightly improve the model's accuracy.
