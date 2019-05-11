# Damage Assessment with Remote Sensing and Social Media (codename: SocialDisaster)
This project provides an interesting method for earthquake damage assessment.

## Process USGS JSON file
It will parse the raw JSON file and generate a numpy array for data analysis and NLP tasks. The data sample is in the dir "USGS"

# Tweets crawler
in the crawl dir run twitter_crawler.py. The data sample is in the dir "Tweets"

## Use of vectorizer and classifier

Run the vectorizer example with "python NLP/vectorizer_test.py"

Then this will save the vectors in the folder NLP/models/vecs.

After that we can run the classifiers with code similar to "python NLP/classifier_test.py" which will read from the vectorizer files in "NLP/models/vecs".

Sequence thens hould be to run vectorizers first and then classifiers

# the prediction error in txt format is in dir "NLP/prediction_error"

# the visual.py in the outer dir is to visualize the data.

# For saving space, only a small amount of data is included in this package


