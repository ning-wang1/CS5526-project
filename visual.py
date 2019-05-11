## Processing the raw data in .json files
import json
from pprint import pprint
import datetime
import time
from helperFunctions import monthName
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit



# Read .json files

def func_powerlaw(x, m, c, c0):
    return c0 + x**m * c

def plot_cdf():
	data = json.load(open('./UsgsData/earthquakes_merged_2009-2019_count19294.json'))
	fea = data["features"]
	eqk_mag = np.zeros(len(fea))

	for i in range(len(fea)):
		eqk_mag[i] = fea[i]["properties"]["mag"]


	yy = eqk_mag
	xx = np.linspace(5,7, num=21)
	plt.figure(figsize=(10,4))

	ax1=plt.subplot(1,2,1)
	#pmf
	weights = np.ones_like(yy)/float(len(yy))
	plt.hist(yy,bins=20, range=(5,7), density=False, facecolor='blue',alpha=0.9,weights=weights)


	#power law fitting
	cnt, bins = np.histogram(yy,bins=21, range=(5,7),density=False,weights=weights)
	popt, pcov = curve_fit(func_powerlaw,xx,cnt/sum(cnt),p0=np.asarray([-1,10,0]))
	print(popt)
	a0 = np.round(popt[0],2)
	a1 = np.round(popt[1],2)
	a2 = np.round(popt[2],2)
	plt.plot(xx, func_powerlaw(xx, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


	plt.xlabel('Magnitude of earthquake')
	plt.ylabel('PMF')

	#cdf
	ax2 =plt.subplot(122)
	ecdf = sm.distributions.ECDF(yy)
	y = ecdf(xx)
	plt.plot(xx,y,color='blue')
	plt.xlabel('Magnitude of earthquake')
	plt.ylabel('CDF')
	plt.show()

	print(eqk_mag)

plot_cdf()

	#eqk_time = fea[i]["properties"]["time"]

	#eqk_place = fea[i]["properties"]["place"]

	#eqk_date = datetime.date.fromtimestamp(int(eqk_time/1000))

	#print('Mag:',eqk_mag,eqk_place,monthName(eqk_date.month),eqk_date.day,eqk_date.year)

def plot_tweets_num():
	#data = json.load(open('./Tweets/Tweets_earthquakes_merged_2009-2019_count19294.json'))
	data = json.load(open('./Tweets/Tweets_earthquake_world_2017_mag5_count1559.json'))
	eqk_num = len(data)
	tweet_num = np.zeros(eqk_num)

	for i in range(eqk_num):
		tweet_num[i] = len(data[i]["tweets"])
	print(tweet_num)


	yy = tweet_num
	xx = np.linspace(0,250, num=251)
	plt.figure(figsize=(8,6))

	ax1=plt.subplot(1,2,1)
	#pmf
	weights = np.ones_like(yy)/float(len(yy))
	plt.hist(yy,bins=250, range=(0,250), density=False, facecolor='blue',alpha=0.5,weights=weights)
	plt.xlabel('Number of crawled tweets for a certain earthquake')
	plt.ylabel('PMF')

	#cdf
	ax2 =plt.subplot(122)
	ecdf = sm.distributions.ECDF(yy)
	y = ecdf(xx)
	plt.plot(xx,y,color='blue')
	plt.xlabel('Number of crawled tweets for a certain earthquake')
	plt.ylabel('CDF')
	plt.show()


plot_tweets_num()

# Store the earthquake information that will be used for training TW
def plot_trend():
	plt.figure()
	num_eqk = np.array([2074,2359,2693,1679,1596,1729,1558,1696,1559,1809])
	year = np.array(range(2009,2019))
	plt.plot(year,num_eqk)
	plt.bar(year,num_eqk)
	plt.xlabel('Year')
	plt.ylabel('Number of Earquakes (mag>5) record by USGS')
	plt.show()