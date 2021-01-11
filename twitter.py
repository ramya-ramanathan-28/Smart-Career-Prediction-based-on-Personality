import tweepy #https://github.com/tweepy/tweepy
import csv

#Twitter API credentials
consumer_key = "2V8tFuEu8F2eiMbNQFt4pUfFB"
consumer_secret = "n2EkO8cvfXeW8AFdRz9XsrEks3EziLsLlWrsXunp9y5bWSJgkY"
access_key = "831571908013412352-wr7pza2d0b0qgu3W1qZguMzmk8AMIqy"
access_secret = "414kzjo4wHO02jJa3vm37pQy56QSEetqGewM4it5sVNzl"


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
		print ("getting tweets before %s" % (oldest))
		
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
	with open('input/%s_tweets.csv' % screen_name, mode='w', encoding='utf-8') as f:
		writer = csv.writer(f)
		writer.writerow(["id","created_at","text"])
		writer.writerows(outtweets)
		print('made file')
	
	pass


if __name__ == '__main__':
	#pass in the username of the account you want to download
	get_all_tweets("LiamPayne")
