import keras
import tensorflow as tf
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"
print("python:{}, keras:{}, tensorflow: {}".format(sys.version, keras.__version__, tf.__version__))

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from sklearn.preprocessing import LabelBinarizer

import re

# Function to clean data ... will be useful later
def post_cleaner(post):
    """cleans individual posts`.
    Args:
        post-string
    Returns:
         cleaned up post`.
    """
    # Covert all uppercase characters to lower case
    post = post.lower() 
    
    # Remove |||
    post=post.replace('|||',"") 

    # Remove URLs, links etc
    post = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', post, flags=re.MULTILINE) 
    # This would have removed most of the links but probably not all 

    # Remove puntuations 
    puncs1=['@','#','$','%','^','&','*','(',')','-','_','+','=','{','}','[',']','|','\\','"',"'",';',':','<','>','/']
    for punc in puncs1:
        post=post.replace(punc,'') 

    puncs2=[',','.','?','!','\n']
    for punc in puncs2:
        post=post.replace(punc,' ') 
    # Remove extra white spaces
    post=re.sub( '\s+', ' ', post ).strip()
    return post

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "input"]).decode("utf8"))

# load dataset
text=pd.read_csv("input/mbti_1.csv" ,index_col='type')
print(text.shape)
print(text[0:5])
print(text.iloc[2])

# One hot encode labels
labels=text.index.tolist()
encoder=LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
labels=encoder.fit_transform(labels)
labels=np.array(labels)
print(labels[50:55])

mbti_dict={0:'ENFJ',1:'ENFP',2:'ENTJ',3:'ENTP',4:'ESFJ',5:'ESFP',6:'ESTJ',7:'ESTP',8:'INFJ',9:'INFP',10:'INTJ',11:'INTP',12:'ISFJ',13:'ISFP',14:'ISFP',15:'ISTP'}


# Clean up posts
# Covert pandas dataframe object to list. I prefer using lists for prepocessing. 
posts=text.posts.tolist()
posts=[post_cleaner(post) for post in posts]

# Count total words
from collections import Counter

word_count=Counter()
for post in posts:
    word_count.update(post.split(" "))

# Size of the vocabulary available to the RNN
vocab_len=len(word_count)
print(vocab_len)
print('len(posts[0]')
print(len(posts[0]))
print(posts[0])

# Create a look up table 
vocab = sorted(word_count, key=word_count.get, reverse=True)
# Create your dictionary that maps vocab words to integers here
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

posts_ints=[]
for post in posts:
    posts_ints.append([vocab_to_int[word] for word in post.split()])

#print(posts_ints[0])
print(len(posts_ints[0]))

posts_lens = Counter([len(x) for x in posts])
print("Zero-length reviews: {}".format(posts_lens[0]))
print("Maximum review length: {}".format(max(posts_lens)))
print("Minimum review length: {}".format(min(posts_lens)))

seq_len = 500
features=np.zeros((len(posts_ints),seq_len),dtype=int)
for i, row in enumerate(posts_ints):
    features[i, -len(row):] = np.array(row)[:seq_len]
print(features[:10])

# Split data into training, test and validation

split_frac = 0.8

num_ele=int(split_frac*len(features))
rem_ele=len(features)-num_ele
train_x, val_x = features[:num_ele],features[num_ele:int(rem_ele/2)+num_ele]
train_y, val_y = labels[:num_ele],labels[num_ele:int(rem_ele/2)+num_ele]

test_x =features[num_ele+int(rem_ele/2):]
test_y = labels[num_ele+int(rem_ele/2):]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))

lstm_size = 256
lstm_layers = 1
batch_size = 256
learning_rate = 0.01
embed_dim=250

n_words = len(vocab_to_int) + 1 # Adding 1 because we use 0's for padding, dictionary started at 1

from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.recurrent import LSTM
'''
in_out_neurons = 16  
hidden_neurons = 256

model = Sequential()
model.add(LSTM(hidden_neurons, return_sequences=True,
               input_shape=(None, in_out_neurons)))

model.add(Dense(in_out_neurons, input_dim=hidden_neurons))  
model.add(Activation("linear"))  
model.compile(loss="mean_squared_error", optimizer="adam")
'''

tsteps = 50
dropout=0.5
model = Sequential()
model.add(LSTM(256, input_shape=(tsteps, 3)))
model.add(Dropout(dropout))
model.add(Dense(16))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.summary()

model.fit(train_x, train_y, batch_size=256, epochs=3, validation_split=0.05)
