from typing import DefaultDict
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.validation import _deprecate_positional_args
import string as st
from sklearn.feature_extraction.text import TfidfVectorizer
import heapq
from sklearn.metrics import mean_squared_error, confusion_matrix
import swifter
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import pickle
from textblob import TextBlob
import nltk
import nltk.corpus
nltk.download('punkt')

def import_data(filepath: str, sheet_id: str):
    """
    reads data file
    drop uneccesary columns
    """
    xls = pd.ExcelFile(filepath)
    # print (xls.sheet_names)
    train = pd.read_excel(xls, sheet_id)

    train = train.drop(columns=['ProductId', 'UserId', 'Time','HelpfulnessNumerator', 'HelpfulnessDenominator' , 'Helpfulness'])
    X_submission_processed = X_submission.drop(columns=['Id', 'ProductId', 'UserId', 'Time','HelpfulnessNumerator', 'HelpfulnessDenominator' , 'Helpfulness' , 'Score'])
    X_submission_processed['text'] = X_submission_processed['Text'] + ' ' +  X_submission_processed['Summary']
    X_submission_processed = X_submission_processed.drop(columns=['Text', 'Summary'])

    df = df.iloc[:,:11]
    # rename them
    df.columns = [] #list of names

    # replacing all missing value with -1
    df.fillna('-1', inplace=True)

    return train

filepath = x
sheet_id = y
import_data(filepath, sheet_id)


newdf = train
newdf['text'] = newdf['Text'] + ' ' + newdf['Summary']
neg_reviews = newdf[newdf['Score'] <= 3]
pos_reviews = newdf[newdf['Score'] >= 4]

print(len(neg_reviews))  #335,426
print(len(pos_reviews))  #1,062,107

#randomize a selection of 10,000 positive and negative reviews to new df data
data_p = pos_reviews.iloc[np.random.randint(1,1062107, 200000), :]
data_n = neg_reviews.iloc[np.random.randint(1,335426, 200000), :]
data = pd.concat([data_p, data_n])
#len(data)

print(data['text'])


### CLEAN STRINGS ###
### remove punctuations 
def remove_punc(string):
    string = str(string).lower()
    a=[word for word in string if word not in st.punctuation]
    return ''.join(a)

#remove stopwords
def process(string):
    stopword = nltk.corpus.stopwords.words('english')
    stopword.remove('not')
    stopword.remove('very')
    stopword.remove('more')
    stopword.remove('we')
    a = [word for word in tokenizer.tokenize(string) if (word in most_freq)]
    return ' '.join(a)




data['text'] = data['text'].apply(remove_punc)
tokenizer = TreebankWordTokenizer()
print(data.head(3))


### used most frequent words and selected the most relevant from find_n_p_words.py ###
#make a dictionary that contains words and corresponding frequ
wordfreq = {}
for sent in data['text']:
    tokens = tokenizer.tokenize(sent)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1

most_freq = heapq.nlargest(5000, wordfreq, key=wordfreq.get)
print(len(most_freq))



#edit text so it only contains common words
data['text'] = data['text'].apply(process)


# Split training set into training and testing set
#test size is size going to test
X_train, X_test, Y_train, Y_test = train_test_split(
        data,
        data['Score'],
        test_size=1/4.0,
        random_state=0
    )

#ngram_range = (1,1) --> 52.12%, RMSE 1.677
#ngram_range = (2,2) --> 


### making a Document Term Matrix (DTM) from most frequent words ###
count_vectorizer = CountVectorizer(decode_error='ignore', lowercase=True, max_df = 0.8)
tfidf = TfidfVectorizer()
train = tfidf.fit_transform(X_train['text'])
test = tfidf.transform(X_test['text'])

print(tfidf.vocabulary_)
print(train)
print('Train size: ',train.shape)
print(test)
print('Test size: ',test.shape)

### assign each word in every review w/ sentiment
vector = TfidfVectorizer(ngram_range = (1,2), min_df=1)
X = vector.fit_transform(data['Text'])
print(tfidf.get_feature_names_out())

### model using regression analysis as classification"
#explain linear:
#using multiple iterations since we want the accuracy to not be at a "local Optima"
model = LinearSVC().fit(train, Y_train)
accuracy_score = model.score(train, Y_train)
print(str(accuracy_score))


means = []
stds = []
folds = 5
C_vals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
for C_val in C_vals:
    f = LinearSVC(kernel='linear', C = C_val)
    scores = model_selection.cross_val_score(f, train, Y_train, cv = folds)
    means.append(np.mean(scores))
    stds.append(np.std(scores) / np.sqrt(folds))
acc = np.array(means)
stderr = np.array(stds)
C_s = np.array(C_vals)

###making submission file

X_submission_processed['text'] = X_submission_processed['text'].swifter.apply(remove_punc)
X_submission_processed['text'] = X_submission_processed['text'].swifter.apply(process)
X_submission_vect = tfidf.transform(X_submission_processed['text'])


Y_test_predictions = model.predict(test)
X_submission['Score'] = model.predict(X_submission_vect)

# Evaluate your model on the testing set
print("RMSE on testing set = ", mean_squared_error(Y_test, Y_test_predictions))

# Plot a confusion matrix
cm = confusion_matrix(Y_test, Y_test_predictions, normalize='true')
sns.heatmap(cm, annot=True)
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Create the submission file
submission = X_submission[['Id', 'Score']]
submission.to_csv("./data/submission.csv", index=False)


