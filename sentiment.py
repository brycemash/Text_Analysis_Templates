import codecs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List

def import_data(filepath: str, sheet_id: str):
    """
    Adapted from preprocessing.py
    reads data file
    drop uneccesary columns
    """
    xls = pd.ExcelFile(filepath)

    # print (xls.sheet_names)
    df = pd.read_excel(xls, sheet_id)

    # takes only 1st 8 columns to include keywords
    df = df.iloc[:,:11]

    # rename them
    df.columns = ['id','quote','nest','c1','c2','c3', 'board', 'date', 'keywords', 'length', 
                  'notes']

    # replacing all missing value with -1
    df.fillna('-1', inplace=True)

    return df


#takes dataframe
#returns dictionary with keywords and average sentiment score
def average_sentiment_keyword(df):
    """
    make  a dictionary with keywords and the scores of the quotes that the keywords are in (list)
    """
    posts = {}
    
    for index, row in df.iterrows():
        current_key = row['keywords']
        
        #search for non-empty keywords
        if current_key != '-1':
            #iterate over keywords
            # Split the keys and clear empty spaces
            keys = current_key.lower().replace(' ', '').split(',')

            for k in keys:
                
                if k not in posts.keys():
                    posts[k] = [row['scores']]
                else:
                    posts[k].append(row['scores'])
            
    print(posts)
    
    return posts


def average_sentiment_code(df):
    """
    make  a dictionary with keywords and the scores of the quotes that the keywords are in (list)
    """
    posts = {}
    
    for index, row in df.iterrows():
        for code in [row['c1'], row['c2'], row['c3']]:
            #search for non-empty keywords
            if code != '-1':
                if code not in posts.keys():
                    posts[code] = [row['scores']]
                else:
                    posts[code].append(row['scores'])
            
    print(posts)
    return posts


"""
visualize the dictionary as an average of the quote scores
either as scatterplot or diverging texts graph
"""

#dictionary with cols keywords, average sentiment score
#graph
def graphScatterPlot(KeywordsAndTheirScores):
    keywords = []
    averageScores = []
    for key in KeywordsAndTheirScores:
        keywords.append(key)
        averageScores.append(np.mean(KeywordsAndTheirScores[key]))
    #my_color = np.where(averageScores >= 0, 'red', 'blue')
    plt.scatter(keywords, averageScores)
    plt.title("Key words and their average scores", loc='left')
    plt.xlabel('Key Words')
    plt.ylabel('Average Scores')
    plt.show(block=False)

# input: dictionary with cols keywords, sentiment scores
# graph a graph diverging text graph of each keyword and the average of their sentimental scores 
def graphDivergingTexts(KeywordsAndTheirScores):
    keys = list(KeywordsAndTheirScores.keys())
    scores = [np.mean(KeywordsAndTheirScores[i]) for i in keys]
    colors = ['red' if i < 0 else 'green' for i in scores]
    plt.figure(figsize=(14,14), dpi= 80)
    plt.hlines(y=keys, xmin=0, xmax=scores)
    for x, y, tex in zip(scores, keys, scores):
        t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if x < 0 else 'left', 
                 verticalalignment='center', fontdict={'color':'red' if x < 0 else 'green', 'size':14})
    plt.yticks(keys, keys, fontsize=12)
    plt.title('Diverging Text Bars of Codes Sentiment '+sheet_id, fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim(-2.5, 2.5)
    plt.show()



def run_sentiment(df):
    """
    makes another column scores that has sentiment scores of the quote column text
    implements graphDivergingTexts() to print visualization of scores
    """
    
    sid = SentimentIntensityAnalyzer()
    df['scores'] = df['quote'].apply(lambda quote: sid.polarity_scores(quote)['compound'])
    new_df = df[df['nest'] == 0]
    return new_df



# file path for data
filepath = 'Dataset.xlsx'
sheet_id = 'All_Data'

# run sentiment analysis on keywords for selected months (corresponds to excel sheet name)
df = import_data(filepath, sheet_id)
new_df = run_sentiment(df)
#KeywordsAndTheirScores = average_sentiment_keyword(new_df)
KeywordsAndTheirScores = average_sentiment_code(new_df)
graphDivergingTexts(KeywordsAndTheirScores)