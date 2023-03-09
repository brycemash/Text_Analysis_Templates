# Text_Analysis_Templates
Templates for analyzing text data.

Network_Template_Louvain.py and sentiment.py follow a spreadsheet with text("quote"), codes(c1,c2,.. etc.), keywords, and themes. 


Network_Template_Louvain.py makes a network bipartite graph and does louvain clustering on it. 

sentiment.py uses vader (https://github.com/cjhutto/vaderSentiment) to score text by positive or negative connotation on a scale of -1 to 1. 

Score_Prediction.py was made for predicting amazon star ratings based on the reviews given for a product. It splits scores (1-5 stars) into positive (>=4) and negatives, randomizes, cleans text, makes a TFIDF, and trains using a simple LinearSVC. 





