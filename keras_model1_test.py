from keras.models import model_from_yaml
import numpy as np
import pandas as pd
import re
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
#from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
data = pd.read_csv('input/new_mbti.csv')
MAX_NB_WORDS = 20000
b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]


def translate_personality(personality):
    '''
    transform mbti to binary vector
    '''
    return [b_Pers[l] for l in personality]
def getPersonality(val):
    Personality = " "
    for item in val:
        print("score =")
        print(item)
        if item[0]>0.035:
            Personality = Personality+"E"
        else:
            Personality = Personality+"I"
        if item[1]>0.1:
            Personality = Personality+"S"
        else:
            Personality = Personality+"N"
        if item[2]>0.1:
            Personality = Personality+"F"
        else:
            Personality = Personality+"T"
        if item[3]>0.5:
            Personality = Personality+"P"
        else:
            Personality = Personality+"J"
    return Personality
    
def translate_back(personality):
    '''
    transform binary vector to mbti personality
    '''
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s
list_personality_bin = np.array([translate_personality(p) for p in data.type])

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk import word_tokenize
# Lemmatize
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

cachedStopWords = stopwords.words("english")

from sklearn.preprocessing import LabelEncoder

unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
lab_encoder = LabelEncoder().fit(unique_type_list)
def pre_process_data(posts, remove_stop_words=True):

    
    list_posts = []
        ##### Remove and clean comments
        
    temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', posts)
    temp = re.sub("[^a-zA-Z]", " ", temp)
    temp = re.sub(' +', ' ', temp).lower()
    if remove_stop_words:
        temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
    else:
        temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
        
    list_posts.append(temp)

    #del data
    list_posts = np.array(list_posts)
    #list_personality = np.array(list_personality)
    return list_posts
f=open("input/tweet_rramesss.txt", "r")
data2 =f.read()
list_posts = pre_process_data(data2, remove_stop_words=True)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(list_posts)
sequences = tokenizer.texts_to_sequences(list_posts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


from sklearn.model_selection import train_test_split
#x_train, x_val, y_train, y_test = train_test_split(sequences, list_personality_bin, test_size=0.3, random_state=0, stratify=list_personality)

# truncate and pad input sequences
max_review_length = 923
#X_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(sequences, maxlen=max_review_length)

max_features = 20000
batch_size = 32

#print('x_train shape:', X_train.shape)
print('x_test shape:', X_test[0].shape)
result = []
loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
score = loaded_model.predict(X_test[0])
result.append(score[0])
print("The result is: ", getPersonality(result))
#print(score)
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
