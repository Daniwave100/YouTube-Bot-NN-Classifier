import nltk
nltk.data.path.append("/Users/danielcanhedo/nltk_data")
from nltk.corpus import stopwords
print(stopwords.words("english"))  # finally works
from nltk.corpus import stopwords # stop works are commonly used words that do not provide value to training model
from nltk.stem import PorterStemmer # porterstemmer converts similar words to one type of word. for example, running, runner becomes run
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import joblib

dataset = pd.read_csv("TwoVideoPlaylistComments__.csv") # imports our training data set
print(dataset.info())

# data cleaning portion ----------------------------------------------------------------

ps = PorterStemmer()
corpus = []

for i in range(len(dataset)):

    comments = re.sub("^a-zA-Z", " ", dataset["Text"][i]) # we are only going to keep a-z A-Z letters
    comments = comments.lower() # converts to lowercase for consistency
    comments = comments.split() # splits every word by space. we now have individual words

    clean_comments = [] # initializes a list to store cleaned comments
    for word in comments:

        if word not in set(stopwords.words('english')): # this for loop will get rid of stop words and apply stemming (explained above)
            stemmed_word = ps.stem(word)
            clean_comments.append(stemmed_word)

    clean_comments = " ".join(clean_comments)
    corpus.append(clean_comments) # we now have a corpus of clean sentences

# we will now convert the sentences to vectors to send through our model

# TFIDF stands for Term-Frequency-Inverse Document Frequency. It measures how important a word is to a document relative to a corpus
# (collection) of documents.
# Term Frequency (TF) - measures how often a word appears in a document
# Inverse Document Frequency (IDF) - measures how unique or rare a word is across all documents

# TF = # Times word occurs in a document / # total number of words in document
# IDF = log((Total # of documents / Number of documents containing the term) + 1)

# Rare words across documents get a high IDF
# Common words appear in many documents so their IDF is lower
# Think of documents as a sentence

# see the chart (where the number is higher, the more important that word is in that document)
#       bird    cat    dog
# D1    0.00    0.52   0.00
# D2    0.00    0.00   0.52
# D3    0.52    0.00   0.00

# same goes for our matrix if we do X = vectorizer.fit_transform(corupus).toarray()

vectorizer = TfidfVectorizer(max_features=20000, min_df=0.01, max_df=0.9) # max number of words, the amount of times for word to occur to be considered, and then max_df gets rif of words that occur mroe than a certain type
X = vectorizer.fit_transform(corpus).toarray()
print(X[0])

y = dataset.iloc[:, 5].values # gets our boolean row and converts to numpy array
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # splits our dataset into training set and test set

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()

Y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()


print(X_train.shape)
print(Y_train.shape)
print()
print(X_test.shape)
print(y_test.shape)

# all of this prints:
# torch.Size([1640, 186])
# torch.Size([1640])
#
# torch.Size([410, 186])
# torch.Size([410])
# all of this means we have 1640+410 = 2050 sentences in the corpus, 186 vectorized features (need to understand this more)

input_size = 186 # the number of nodes we have
output_size = 2 # we are predicting whether true or false so it is 2
hidden_size = 500 # this can change but lets try with 500

# let's now build the neural network ourselves like we learned in datacamp
model = nn.Sequential(
    nn.Linear(input_size, hidden_size), # builds our first layer. remember they must link with the same hidden_size
    nn.ReLU(), # introduces non-linearity so it can learn more complex relationships (look more into this)
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size),
    nn.LogSoftmax(dim=1)) # we use the softmax activation function which is for classification

# the optimizer is what changes the weights and biases during training so it can learn better, this minimizes the loss function
# it improves the accuracy of the neural network by using gradient descent
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# the loss function is what holds the value for how bad the model performaed. it calculates the difference between the model's
# prediction and the actual label
loss_fn = nn.NLLLoss()

# now we will actually train the neural network
epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()
    Y_pred = model(X_train)
    loss = loss_fn(Y_pred, Y_train) # loss function which computes how different predictions are from actual label
    loss.backward() # backward pass to optimize
    optimizer.step() # optimizes the weights to minimize loss
    print("Epoch", epoch, "Loss:", loss.item())

# so perhaps we can find a way to combine our text classifier with a DecisionTreeRegressor which would take
# into account like count, reply count, etc.

# Save the vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')
torch.save(model.state_dict(), 'model_weights.pth')
# Save entire model
torch.save(model, 'full_model.pt')



