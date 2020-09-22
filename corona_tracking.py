# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 03:42:41 2020

@author: sinha
"""
from matplotlib import pyplot as plt
import OpenBlender
import pandas as pd
import json
from datetime import datetime
import numpy as np
import time
import matplotlib.dates as mdates

'''
-------------------------------------------
GATHERING COVID-19 CONFIRMED CASES
-------------------------------------------
'''
START_DATE = "2020-01-01"
END_DATE = "2020-03-22"
interest_countries = ['US', 'China', 'India', 'Canada', 'Korea', 'Italy', 'France', 'Germany', 'Spain', 'Australia']


action_data = 'API_getObservationsFromDataset'

parameters_data = {'token': '5e7715d195162926d7d221781NCyhgI11wcu4EAoTI6jeMBGXBlkNt',
              'id_dataset' : '5e6ac97595162921fda18076',
              'date_filer': {"start_date": START_DATE + "T06:00:00.000Z",
                             "end_date": END_DATE + "T06:00:00.000Z"},
            'consumption_confirmation': 'on'}

blender_covid_confirmed_data = json.dumps(OpenBlender.call(action_data, parameters_data)['sample'])
confirmed_df = pd.read_json(blender_covid_confirmed_data, convert_dates = False, convert_axes = False)
confirmed_df.to_csv('confirmed_cases.csv', index = False)
'''
Sort data based on timestamp
Reset the index to normal range
Display first 10 observations
'''
confirmed_df = confirmed_df.sort_values('timestamp', ascending = False)
confirmed_df.reset_index(drop = True, inplace=True)

def convertTimestampToDate(df):
    dates = []
    for i in range(len(df['timestamp'])):
        date = datetime.utcfromtimestamp(df['timestamp'][i]).strftime('%d-%m-%Y')
        dates.append ( date )
    return dates

def getTimestampFromDate(df):
    ts = []
    unique_dates = df.dates.unique()
    for date in unique_dates:
        element = datetime.strptime(date, "%d-%m-%Y").timetuple()
        timestamp = time.mktime(element)
        ts.append( timestamp )
    
    return ts

def splitDataByCategory(df):
    unique_dates = df.dates.unique()
    countries_dict = {}
    countries_dict['dates'] = []
    countries_dict['country'] = []
    countries_dict['confirmed_cases'] = []
    countries_dict['deaths'] = []
    countries_dict['recovered'] = []
    
    
    '''
    Lists to populate data for each category
    Size of each list = 60 x 10 - # of unique_dates X # of countries
    '''
    confirmed_cases = np.zeros((len(unique_dates), len(interest_countries)))
    deaths = np.zeros((len(unique_dates), len(interest_countries)))
    recoveries = np.zeros((len(unique_dates), len(interest_countries)))
    
    
    '''
    Maintaining a dictionary to see that same province data is not repeatedly
    added for the same date, only one data is taken for that date for the particular province
    '''
    seen_provinces = {}
    for date in unique_dates:
        seen_provinces[date] = []
    
    '''
    Populate the lists
    '''
    for dt_indx in range(len(unique_dates)):
        dt = unique_dates[dt_indx]  # Get the unique date
        for indx in range(len(df)):
            curr_dt = df['dates'][indx] # Get the current date from the major dataset
            if curr_dt == dt:   # check if the unique date is the same as the current date
                country = df['countryregion'][indx]
                province = df['provincestate'][indx]
                
                # Checking if the country at this index in the major dataset is to be considered
                if country in interest_countries:
                    #Check if the province for this date has not already be seen earlier
                    if province not in seen_provinces[dt]:
                        seen_provinces[dt].append(province)
                        try:
                            # Populating the lists
                            confirmed_cases[dt_indx, interest_countries.index(country)] += int(df['confirmed'][indx])
                            deaths[dt_indx, interest_countries.index(country)] += int(df['deaths'][indx])
                            recoveries[dt_indx, interest_countries.index(country)] += int(df['recovered'][indx])
                        except:
                            confirmed_cases[dt_indx, interest_countries.index(country)] += 0
                            deaths[dt_indx, interest_countries.index(country)] += 0
                            recoveries[dt_indx, interest_countries.index(country)] += 0
    
    '''
    Storing each category's data in separate dictionaries, with each country's data
    based on the date
    
    These dictionaries will finally be converted to the dataframe
    '''                    
    confirmed_cases_dict = {}
    confirmed_cases_dict['dates'] = []
    deaths_dict = {}
    deaths_dict['dates'] = []
    recoveries_dict = {}
    recoveries_dict['dates'] = []
    
    '''
    Initialize the columns of the dictionary
    '''
    for country in interest_countries:
        confirmed_cases_dict['confirmed_' + country] = []
        deaths_dict['deaths_' + country] = []
        recoveries_dict['recovered_' + country] = []
    
    
    '''
    Populate the dictionaries by copying the data from the list
    '''
    for dt_indx in range(len(unique_dates)):
        confirmed_cases_dict['dates'].append( unique_dates[dt_indx] )
        deaths_dict['dates'].append( unique_dates[dt_indx] )
        recoveries_dict['dates'].append( unique_dates[dt_indx] )
    
        for country_indx in range(len(interest_countries)):
            country = interest_countries[country_indx]
            
            countries_dict['dates'].append( unique_dates[dt_indx] )
            countries_dict['country'].append( country )
            countries_dict['confirmed_cases'].append( confirmed_cases[dt_indx, country_indx] )
            countries_dict['deaths'].append( deaths[dt_indx, country_indx] )
            countries_dict['recovered'].append( recoveries[dt_indx, country_indx] )
            
            confirmed_cases_dict['confirmed_' + country].append( confirmed_cases[dt_indx, country_indx] )
            deaths_dict['deaths_' + country].append( deaths[dt_indx, country_indx] )
            recoveries_dict['recovered_' + country].append( recoveries[dt_indx, country_indx] )

    '''
    Convert the dictionaries to dataframes for each category, organized date-wise
    '''            
    country_data = pd.DataFrame(countries_dict)
    confirmed_data = pd.DataFrame(confirmed_cases_dict)
    deaths_data = pd.DataFrame(deaths_dict)
    recoveries_data = pd.DataFrame(recoveries_dict)
    
    return country_data, confirmed_data, deaths_data, recoveries_data

def plotData(df, keyword, title, ylab):
    ax = df.plot(x = 'dates', 
                         y = [col for col in df.columns if keyword in col], 
                         figsize = (15, 7), rot=45,
                         title = title)
    xlab = [item.get_text()[:5] for item in ax.get_xticklabels()]
    _ = ax.set_xticklabels(xlab)
    ax.set_xlabel("Dates")
    ax.set_ylabel(ylab)
    plt.show()


confirmed_df['dates'] = convertTimestampToDate(confirmed_df)
country_data, confirmed_data, deaths_data, recoveries_data = splitDataByCategory(confirmed_df)
ts = getTimestampFromDate(confirmed_df)

confirmed_data['timestamp'] = ts
deaths_data['timestamp'] = ts
recoveries_data['timestamp'] = ts

confirmed_data = confirmed_data.sort_values('timestamp')
deaths_data = deaths_data.sort_values('timestamp')
recoveries_data = recoveries_data.sort_values('timestamp')

plotData(confirmed_data, 'confirmed', "Confirmed COVID-19 Cases so far", "Confirmed Cases")
plotData(deaths_data, 'deaths', "Confirmed Deaths due to COVID-19 Cases so far", "Reported Deaths")
plotData(recoveries_data, 'recovered', "Recovered COVID-19 Cases so far", "Recovered cases")


'''
-------------------------------------------
GATHERING NEWS OF COVID-19
[Sources : ABC NEWS, WALL STREET JOURNAL, CNN NEWS AND USA TODAY TWITTER] 
-------------------------------------------
'''
action_news = 'API_getOpenTextData'

parameters_news = {'token': '5e7715d195162926d7d221781NCyhgI11wcu4EAoTI6jeMBGXBlkNt',
              'consumption_confirmation': 'on',
              'date_filter': {"start_date": START_DATE + "T06:00:00.00Z",
                              "end_date": END_DATE + "T06:00:00.000Z"},
            'sources': [
                    #Wall Street Journal
                    {'id_dataset': '5e2ef74e9516294390e810a9',
                     'features': ['text']},
                     
                     #ABC News Headlines
                    {'id_dataset': '5d8848e59516294231c59581',
                     'features': ['headline','title']},
                     
                     #USA Today Twiiter
                     {'id_dataset': '5e32fd289516291e346c1726',
                     'features': ['text']},
                      
                      #CNN News
                      {'id_dataset': '5d571b9e9516293a12ad4f5c',
                     'features': ['headline','title']}
                     ],
            'aggregate_in_time_interval': {'time_interval_size': 24*60*60},
            'text_filter_search': ['covid', 'coronavirus', 'ncov']
            }

blender_downloaded_news_data = json.dumps(OpenBlender.call(action_news, parameters_news)['sample'])
news_df = pd.read_json(blender_downloaded_news_data, convert_dates = False, convert_axes = False)
news_df.to_csv('news_collected.csv', index = False)

'''
Sort news data based on timestamp
Reset the index to normal range
Display first 10 observations
'''
news_df = news_df.sort_values('timestamp', ascending = False)
news_df.reset_index(drop = True, inplace = True)

'''
-------------------------------------------
AGGREGATE THE NEWS DATA FOR EACH OF THE COUNTRIES
-------------------------------------------
'''
for country in interest_countries:
    news_df['count_news_' + country] = [len([text for text in daily_lst if country.lower() in text]) for daily_lst in news_df['source_lst']]

news_df.reindex(index = news_df.index[::-1])
news_y_data = [col for col in news_df.columns if 'count' in col]
news_df.plot(x = 'timestamp', y = news_y_data, figsize = (20, 7), kind = 'area')


'''
DRAW WORDCLOUD
'''
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
plt.figure()
plt.imshow(WordCloud(max_font_size=50, max_words=80, background_color="white").generate(' '.join([val for val in news_df['source'][0: 20]])), interpolation="bilinear")
plt.axis("off")
plt.show()


'''
-------------------------------------------
GATHERING OTHER AFFECTED INDICATORS
[INDICATORS : Exchange Rates(Euro, Pound, Rupee), Material Prices(Cude Oil, Corn, Platinum, Tin), Stocks(Coca Cola, Dow Jones)] 
-------------------------------------------
'''
action_prices = 'API_getObservationsFromDataset'

parameters_prices = {
 'token':'5e7715d195162926d7d221781NCyhgI11wcu4EAoTI6jeMBGXBlkNt',
 'id_dataset':'5d4c14cd9516290b01c7d673',
 'aggregate_in_time_interval':{"output":"avg","empty_intervals":"impute","time_interval_size":86400},
 'blends':[
        #Yen vs USD              
{"id_blend":"5d2495169516290b5fd2cee3","restriction":"None","blend_type":"ts","drop_features":[]},
        # Euro Vs USD
{"id_blend":"5d4b3af1951629707cc1116b","restriction":"None","blend_type":"ts","drop_features":[]},
        # Pound Vs USD              
{"id_blend":"5d4b3be1951629707cc11341","restriction":"None","blend_type":"ts","drop_features":[]},
        # Corn Price    
{"id_blend":"5d4c23b39516290b01c7feea","restriction":"None","blend_type":"ts","drop_features":[]},
        # CocaCola Price     
{"id_blend":"5d4c72399516290b02fe7359","restriction":"None","blend_type":"ts","drop_features":[]},
        # Platinum price             
{"id_blend":"5d4ca1049516290b02fee837","restriction":"None","blend_type":"ts","drop_features":[]},
        # Tin Price
{"id_blend":"5d4caa429516290b01c9dff0","restriction":"None","blend_type":"ts","drop_features":[]},
        # Crude Oil Price
{"id_blend":"5d4c80bf9516290b01c8f6f9","restriction":"None","blend_type":"ts","drop_features":[]}],
'date_filter':{"start_date":START_DATE + "T06:00:00.000Z","end_date":END_DATE + "T06:00:00.000Z"},
'consumption_confirmation':'on' 
}
commodities_prices = json.dumps(OpenBlender.call(action_prices, parameters_prices)['sample'])
prices_data = pd.read_json(commodities_prices, convert_dates=False, convert_axes=False)
prices_data = prices_data.sort_values('timestamp', ascending=False)
prices_data.reset_index(drop=True, inplace=True)


'''
Compress and normalize data into [0,1] range
'''
prices_data.dropna(0)
compressed_data = prices_data.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).apply(lambda x: (x - x.min())/(x.max()-x.min()))
compressed_data['timestamp'] = prices_data['timestamp']

'''
Select columns of interest
'''
cols_of_interest = ['timestamp', 'PLATINUM_PRICE_price', 'CRUDE_OIL_PRICE_price', 'COCACOLA_PRICE_price', 'open', 
                    'CORN_PRICE_price', 'TIN_PRICE_price', 'PLATINUM_PRICE_price']
compressed_data = compressed_data[cols_of_interest]
compressed_data.rename(columns = {'open': 'DOW_JONES_price'}, inplace = True)

# Rearranging data based on dates
compressed_data['dates'] = convertTimestampToDate(compressed_data)
compressed_data = compressed_data.drop('timestamp', axis=1)
compressed_data['timestamp'] = getTimestampFromDate(compressed_data)
compressed_data = compressed_data.sort_values('timestamp')

'''
Plot this data
'''
from matplotlib import pyplot as plt
fig_pr, ax_pr = plt.subplots(figsize = (17,7))
plt = compressed_data.plot(x = 'dates', 
                           y = ['PLATINUM_PRICE_price', 'CRUDE_OIL_PRICE_price', 'COCACOLA_PRICE_price',
                                'DOW_JONES_price', 'CORN_PRICE_price', 'TIN_PRICE_price', 'PLATINUM_PRICE_price'],
                        ax = ax_pr)
plt.show()
