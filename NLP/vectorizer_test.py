from Vectorizer import Vectorizer
import os

working_dir = os.path.dirname(os.path.realpath(__file__))

vect = Vectorizer('tfidf_world_2019','tfidf')
vect.vectorize(working_dir + "/../Tweets/Tweets_earthquake_world_2019_mag5_count506.json")

#vect = Vectorizer('tfidf_world_2018','tfidf')
#vect.vectorize(working_dir + "/../Tweets/Tweets_earthquake_world_2018_mag5_count1809.json")

#vect = Vectorizer('tfidf_world_2009-2019','tfidf')
#vect.vectorize(working_dir + "/../Tweets/Tweets_earthquakes_merged_2009-2019_count19294.json")

#vect = Vectorizer('w2v_world_2019','w2v')
#vect.vectorize(working_dir + "/../Tweets/Tweets_earthquake_world_2019_mag5_count506.json")

#vect = Vectorizer('w2v_world_2018_skipgram','w2v_skipgram')
#vect.vectorize(working_dir + "/../Tweets/Tweets_earthquake_world_2018_mag5_count1809.json")

#vect = Vectorizer('w2v_world_2018_cbow','w2v_cbow')
#vect.vectorize(working_dir + "/../Tweets/Tweets_earthquake_world_2018_mag5_count1809.json")

#vect = Vectorizer('w2v_world_2017_cbow','w2v_cbow')
#vect.vectorize(working_dir + "/../Tweets/Tweets_earthquake_world_2017_mag5_count1559.json")

#vect = Vectorizer('w2v_world_2009-2019','w2v')
#vect.vectorize(working_dir + "/../Tweets/Tweets_earthquakes_merged_2009-2019_count19294.json")



vect.save_model()
print("Finished vectorizing")


