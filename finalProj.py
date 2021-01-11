from flask import Flask, render_template, request, json, session, url_for, redirect, flash
app = Flask(__name__)



import tweepy #https://github.com/tweepy/tweepy
import csv
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk import word_tokenize
from keras.models import model_from_yaml
import pandas as pd
import re
import itertools as it
import os
import coll
feed=0
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
yaml_file = open('model_new.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model_new.h5")
#print("Loaded model from disk")
data = pd.read_csv('input/new_mbti.csv')
MAX_NB_WORDS = 20000

yaml_file2 = open('modelk1.yaml', 'r')
loaded_model_yaml2 = yaml_file2.read()
yaml_file2.close()
loaded_model2  = model_from_yaml(loaded_model_yaml2)
# load weights into new model
loaded_model2.load_weights("modelk1.h5")
#print("Loaded model from disk")

yaml_file3 = open('model_t.yaml', 'r')
loaded_model_yaml3 = yaml_file3.read()
yaml_file3.close()
loaded_model3  = model_from_yaml(loaded_model_yaml3)
# load weights into new model
loaded_model3.load_weights("model_t.h5")
print("Loaded model from disk")

b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]

unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
# Lemmatize
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

cachedStopWords = stopwords.words("english")

from sklearn.preprocessing import LabelEncoder

unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
lab_encoder = LabelEncoder().fit(unique_type_list)



#mbti_dict={0:'ENFJ',1:'ENFP',2:'ENTJ',3:'ENTP',4:'ESFJ',5:'ESFP',6:'ESTJ',7:'ESTP',8:'INFJ',9:'INFP',10:'INTJ',11:'INTP',12:'ISFJ',13:'ISFP',14:'ISFP',15:'ISTP'}

def translate_personality(personality):
    '''
    transform mbti to binary vector
    '''
    return [b_Pers[l] for l in personality]

def getPersonality(val1, val2, val3):
    Personality = " "
    for item in val1:
        #print("score =")
        #print(item)
        #if item[0] <= 0.043 or (item[0] >= 0.11 and item[0] < 0.15) or (item[0]>=0.22 and item[0]< 0.25 ) :
        if item[0] >0.23:
            Personality = Personality+"I"
        else:
            Personality = Personality+"E"
        #if item[1] < 0.008 or (item[1] > .017 and item[1] < 0.024) or (item[1] > .064 and item[1] < 0.13) or (item[1] > 0.29 and  item[1]<0.33):
        if abs(item[1]/0.35)<0.5:
            Personality = Personality+"S"
        else:
            Personality = Personality+"N"
    for item in val2:
        #print("score =")
        #print(item)
        if item[2]>0.3:   #0.4
            Personality = Personality+"T"
        else:
            Personality = Personality+"F"        
    for item in val2:
        #print("score =")
        #print(item)
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

def read_data(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

#Twitter API credentials
consumer_key = "2V8tFuEu8F2eiMbNQFt4pUfFB"
consumer_secret = "n2EkO8cvfXeW8AFdRz9XsrEks3EziLsLlWrsXunp9y5bWSJgkY"
access_key = "831571908013412352-wr7pza2d0b0qgu3W1qZguMzmk8AMIqy"
access_secret = "414kzjo4wHO02jJa3vm37pQy56QSEetqGewM4it5sVNzl"

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
    post=re.sub("@\\w+ *", "", post)
    post=re.sub("#\\w+ *", "", post)
    post= re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '',post)
    post=re.sub("[^\x00-\x7F]+","", post)
    post=re.sub("^([x][f][0]).* $","", post)
    post=post.replace("xf0x9fx98x9exf0x9fx98xa2", " ")
    post=post.replace("xf0x9fx8dxb0xe2x98x95xf0x9fx8dxa6xf0x9fx8dx9exf0x9fx98x8axf0x9fx8ex82xf0x9fx91x8cxf0x9fx91x8dxe2x9cx8cb", " ")
    post=post.replace("xe2x80xa6", " ")
    post=post.replace("xf0x9fx98x82", " ")
    post=post.replace("xe2x80x9c", " ")
    # Remove puntuations 
    puncs1=['@','#','$','%','^','&','*','(',')','-','_','+','=','{','}','[',']','|','\\','"',"'",';',':','<','>','/']
    for punc in puncs1:
        post=post.replace(punc,'') 

    puncs2=[',','.','?','!','\n']
    for punc in puncs2:
        post=post.replace(punc,' ') 
    # Remove extra white spaces
    post=re.sub( '\s+', ' ', post ).strip()             #this is the unicode removal
    return post

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

def predict(screen_name):
        cachedStopWords = stopwords.words("english")
        unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
        lab_encoder = LabelEncoder().fit(unique_type_list)
        f=open("input/%s_tweets.txt" % screen_name, "r")
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
        max_review_length = 600
        #X_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
        X_test = sequence.pad_sequences(sequences, maxlen=max_review_length)

        max_features = 20000
        batch_size = 32


        print('x_test shape:', X_test[0].shape)
        result = []
        result2 = []
        result3 = []
        loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        score = loaded_model.predict(X_test)
        loaded_model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        score2 = loaded_model2.predict(X_test)
        result2.append(score2[0])
        result.append(score[0])
        loaded_model3.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        score3 = loaded_model3.predict(X_test)
        result3.append(score3[0])
        pType = getPersonality(result, result2, result3)
        print("The result is: ", pType)
        return (pType)

@app.route("/career/<screen_name>", methods = ['POST', 'GET'])
def career(screen_name):
        pType = predict(screen_name)
        print (pType[1:])
        pType = pType[1:]
        #time.sleep(5)
        #ptype = coll.career(pType[1:])
        return redirect(url_for('results', pType=pType))

@app.route("/tweets/<screen_name>", methods = ['POST', 'GET'])
def get_all_tweets(screen_name):
        #Twitter only allows access to a users most recent 3240 tweets with this method
        
        #authorize twitter, initialize tweepy
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_key, access_secret)
        api = tweepy.API(auth)
        
        #initialize a list to hold all the tweepy Tweets
        alltweets = []  
        
        #make initial request for most recent tweets (200 is the maximum allowed count)
        new_tweets = api.user_timeline(screen_name = screen_name,count=50)
        
        #save most recent tweets
        alltweets.extend(new_tweets)
        
        #save the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        
        #keep grabbing tweets until there are no tweets left to grab
        while len(new_tweets) > 0:
                #print ("getting tweets before %s" % (oldest))
                
                #all subsiquent requests use the max_id param to prevent duplicates
                new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
                
                #save most recent tweets
                alltweets.extend(new_tweets)
                
                #update the id of the oldest tweet less one
                oldest = alltweets[-1].id - 1
                
                print ("...%s tweets downloaded so far" % (len(alltweets)))
        
        #transform the tweepy tweets into a 2D array that will populate the csv 
        outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]
        
        #write the csv  
        with open('input/%s_tweets.csv' % screen_name, mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(["id","created_at","text"])
                writer.writerows(outtweets)
                print('made file')
        
        pass
        tweets = pd.read_csv('input/%s_tweets.csv' % screen_name, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
        text=""
        for i in range (1,tweets.shape[0]):
                if str( tweets["text"][i] )[2:-1][0:4]=="RT @":
                        print ('retweet')
                        continue
                text=text+ str( tweets["text"][i] )[2:-1]
                print(str( tweets["text"][i] )[2:-1])
                print('\n')

        print(post_cleaner(text))
    
        with open('input/%s_tweets.txt' % screen_name, "w") as x:
            for i in post_cleaner(text):
                x.write(i)
        return redirect(url_for('career', screen_name = screen_name))
        

@app.route("/essay", methods = ['POST', 'GET'])
def essay():
    print ('2')
    if request.method == 'POST' :
        if 'path' in request.form:
            global feed
            feed=0
            path = str(request.form['path'])
            if (os.path.isfile(path))!=True:
                return render_template("essay.html")
            x = read_data(path)
            new = []
            for i in x:
                new += post_cleaner(i)
            with open ('input/essay_tweets.txt', 'w') as f:
                for i in x:
                    f.write(i)
            return redirect(url_for('career', screen_name = 'essay'))
    return render_template("essay.html")
    

@app.route("/twitter", methods = ['POST', 'GET'])
def twitter():
    if request.method == 'POST' :
        if 'twitter' in request.form:
            global feed
            feed=0
            handle = str(request.form['twitter'])
            print (handle)
            return redirect(url_for('career', screen_name = handle))
    print ('1')
    return render_template("twitter.html")

@app.route("/results/<pType>", methods = ['POST', 'GET'])
def results(pType):

    careers = coll.careerRet(pType)
    print (careers)
    return render_template("viewResults.html",pType=pType, data=careers)

@app.route("/feedback/<pType>", methods = ['POST', 'GET'])
def feedback(pType):
    global feed
    if (feed==1):
        return redirect(url_for('feedback_completed', pType=pType))
    careers = coll.careerRet(pType)
    print (careers)
    if request.method == 'POST' :
        print("POST2")
        feedback = []
        print (request.form)
        if 'myRange' in request.form:
            print("INSIDE")
            feedback = feedback + [int(request.form['myRange'])]
            print (feedback[0])
        if 'myRange1' in request.form:
            print("INSIDE")
            feedback = feedback + [int(request.form['myRange1'])]
            print (feedback[1])
        if 'myRange2' in request.form:
            print("INSIDE")
            feedback = feedback + [int(request.form['myRange2'])]
            print (feedback[2])
        if 'myRange3' in request.form:
            print("INSIDE")
            feedback = feedback + [int(request.form['myRange3'])]
            print (feedback[3])
        coll.career(pType, feedback)
        return redirect(url_for('feedback_completed', pType=pType))
    return render_template("feedback.html",pType=pType, data=careers)

     

@app.route("/feedback-completed/<pType>", methods = ['POST', 'GET'])
def feedback_completed(pType):
    global feed
    feed = 1
    return render_template("feedback-completed.html",pType=pType)

@app.route("/", methods = ["POST", "GET"])
def index():
        return redirect(url_for("twitter"))

if __name__ == '__main__':
        app.run()
        
