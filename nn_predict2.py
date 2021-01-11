import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import re

data = pickle.load( open( "old_training_data", "rb" ) )
test_x = data['test_x']
test_y = data['test_y']
#posts = data['posts']
labels = data['labels']
#features = data['features']
vocab = data['vocab']
#vocab_len = data['vocab_len']
vocab_to_int = data['vocab_to_int']

print (test_x.shape)
def read_data(filename):
  with open(filename) as f:
    content = f.readlines()
  content = [x.strip() for x in content]
  return content

def post_cleaner(post):
    """cleans individual posts`.
    Args:
        post-string
    Returns:
         cleaned up post`.
    """
    print (post)
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
#input_name = "input"


#if len(FLAGS.checkpoint_dir) == 0:
#        FLAGS.checkpoint_dir = xdg.save_data_path(os.path.join('deepspeech','checkpoints',FLAGS.importer))
mbti_dict={0:'ENFJ',1:'ENFP',2:'ENTJ',3:'ENTP',4:'ESFJ',5:'ESFP',6:'ESTJ',7:'ESTP',8:'INFJ',9:'INFP',10:'INTJ',11:'INTP',12:'ISFJ',13:'ISFP',14:'ISFP',15:'ISTP'}
'''
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "input"]).decode("utf8"))
'''

#text = read_data("tweet.txt")
#f=open("C:/Users/A/Desktop/selfStudySemVI/input/tweet.txt", "r")
text=read_data("input/tweet.txt")
#text="Hello how are you"
print (text)
posts=[post_cleaner(texts) for texts in text]
print('ram rohan r')
print (posts)
for post in posts:
    print('rohan')
    print(post)
#posts=[post_cleaner(post) for post in posts]
#posts2 = []
#posts2 = [posts2.append(post) for post in posts]
#posts = posts2
# Count total words
from collections import Counter
word_count=Counter()
for post in posts :
  word_count.update(post.split(" "))
   
# Create a look up table 
#vocab = sorted(word_count, key=word_count.get, reverse=True)
# Create your dictionary that maps vocab words to integers here
#vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

vocab = sorted(word_count, key=word_count.get, reverse=True)
# Create your dictionary that maps vocab words to integers here
vocab_to_int2 = {word: ii for ii, word in enumerate(vocab, 1)}
'''
posts_ints=[]
for post in posts :
  for word in post.split():
    try:
      posts_ints.append(vocab_to_int[word])
    except KeyError:
      continue
posts_lens = Counter([len(x) for x in posts])
'''
posts_ints=[]
for post in posts:
    posts_ints.append([vocab_to_int2[word] for word in post.split()])
seq_len = 500
features=np.zeros((len(posts_ints),seq_len),dtype=int)
for i, row in enumerate(posts_ints):
    features[i, -len(row):] = np.array(row)[:seq_len]
'''
seq_len = 500
features=np.zeros((len(posts),seq_len),dtype=int)
for row in enumerate(posts):
    features[-len(row):] = np.array(row)[:seq_len]

for i, row in enumerate(posts):
    features[i, -len(row):] = np.array(row)[:seq_len]
'''
print(features[:])

print (features.shape)

lstm_size = 256
lstm_layers = 1
batch_size = 256
learning_rate = 0.01
embed_dim=250

n_words = len(vocab_to_int) + 1 # Adding 1 because we use 0's for padding, dictionary started at 1
print ('1')
# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    input_data = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    print ('2')


# Embedding
with graph.as_default():
    #embedding= tf.Variable(tf.random_uniform(shape=(n_words,embed_dim),minval=-1,maxval=1))
    embedding= tf.Variable(tf.random_uniform(shape=(n_words,embed_dim),minval=-1,maxval=1))
    embed=tf.nn.embedding_lookup(embedding,input_data)
    print(embed.shape)

#LSTM cell
with graph.as_default():
    # basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob=keep_prob)
    
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop]* lstm_layers)
    
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)
    print ('3')


with graph.as_default():
    outputs,final_state=tf.nn.dynamic_rnn(cell,embed,dtype=tf.float32 )
    print ('4')


with graph.as_default():
    
    pre = tf.layers.dense(outputs[:,-1], 16, activation=tf.nn.relu)
    predictions=tf.layers.dense(pre, 16, activation=tf.nn.softmax)
    
    cost = tf.losses.mean_squared_error(labels_, predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    print ('5')


with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print ('6')


def get_batches(x, batch_size=100):
    
    n_batches = len(x)//batch_size
    x = x[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size]

with graph.as_default():
    saver = tf.train.Saver()

test_acc = []
with tf.Session(graph=graph) as sess:
    print ('6.5')
    new_saver = tf.train.import_meta_graph('checkpoints/mbti.ckpt.meta')
    #new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    new_saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    print ('7')
    '''
    for ii, (x, y) in enumerate(get_batches(posts, batch_size), 1):
        feed = {input_data: x[0:1],
                keep_prob: 0.8,}
        res = sess.run(predictions, feed_dict=feed)
        print ('8')

        print (res)
        maxi = 0
        for j in range(0, batch_size):-
            maxi = argmax(res[j])
            result = mbti_dict[maxi]
            print ('label : ', labels[6940+867+j*16+i])
            print("result : ", result)
    '''
    #print (test_x[0])
    print (features[0])
    feed = {input_data : [features[0]], keep_prob : 0.8}
    res = sess.run(predictions, feed_dict=feed)
    print (res)
    '''
    feed = {input_data : features, keep_prob : 0.8}
    res = sess.run(predictions, feed_dict=feed)
    print (res[0])
    '''
    maxi=0
    i=0
    s=0
    for item in res[0]:
      s=s+item
      if item>maxi:
        maxi=item
        pos=i
      i=i+1
        
    #print(tf.argmax(res, 1))
    
    
    #maxi = tf.cast(tf.argmax(res, 1), dtype='int')
    result = mbti_dict[pos]
    #print (labels[6940])
    print("result : ", result)
    print(s)
    print(features)
    #nn_output = sess.run(outputs , feed_dict=feed)
    #print (nn_output)
    
    #nn_output = sess.run(outputs , feed_dict=feed)
    #print (nn_output)
    

'''
    #nn_output = sess.run(outputs , feed_dict={ input_data: x[0], keep_prob: 0.8 })
    #y = tf.nn.softmax_cross_entropy()
    #feed_dict = {input : features[0]}
    #predictions = sess.run([features[0]], feed_dict = {input_data: features[0]})
    #print ('predictions', predictions)
    #print ("predictions", predictions.eval(feed_dict={input_data: features}, session=sess))
'''
