import pandas as pd
import re

tweets = pd.read_csv("input/LiamPayne_tweets.csv", sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
text=""
for i in range (1,tweets.shape[0]):
    if str( tweets["text"][i] )[2:-1][0:4]=="RT @":
        print ('retweet')
        continue
    text=text+ str( tweets["text"][i] )[2:-1]
    print(str( tweets["text"][i] )[2:-1])
    print('\n')

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
    post=re.sub( '\s+', ' ', post ).strip()		#this is the unicode removal
    return post


print(post_cleaner(text))
    
with open("input/tweet_liam.txt", "w") as x:
    for i in post_cleaner(text):
        x.write(i)
