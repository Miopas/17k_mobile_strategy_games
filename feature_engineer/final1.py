#import jieba as jieba
#import nltk as nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#from sns import countplot
import datetime
import sys

inputfile = sys.argv[1]
outputfile = sys.argv[2]

#data = pd.read_csv("appstore_games.csv")
data = pd.read_csv(inputfile)

data = data.dropna(axis=0,subset=['Average User Rating']) # drop the rows which target is none

developers = data['Developer'].value_counts()  # calculate the developers of games
pd.set_option('display.max_rows', None)
#print(data['Developer'].value_counts())
data = data.drop(['URL', 'Icon URL','Description','Developer','Primary Genre'], axis=1)  #URL and ID are unrelated to the rating, delete 'Icon URL'(temporarily)
n,m = data.shape
print(n,m)

# Name, word cut
# use fencitest2.py to calculate the frequency of the word, the top 3 are game, puzzle, free, war

# calculate the top3 key words in the games whose rating more than 3.5
higher_rating_games = data[data['Average User Rating'] >3.5]
#print(higher_rating_games)
#higher_rating_games.to_csv('E:\phd tools\ML_Problem2\higher_rating_games.csv')

"""for i in range(n):
    if()"""

"""fig, ax = plt.subplots(1, 2, figsize=(16,32))
wordcloud = WordCloud(background_color='white',width=800, height=800).generate(' '.join(higher_rating_games['Name']))
wordcloud_sub = WordCloud(background_color='white',width=800, height=800).generate(' '.join(higher_rating_games['Subtitle'].dropna().astype(str)) )
ax[0].imshow(wordcloud)
ax[0].axis('off')
ax[0].set_title('Wordcloud(Name)')
ax[1].imshow(wordcloud_sub)
ax[1].axis('off')
ax[1].set_title('Wordcloud(Subtitle)')
plt.show()"""

"""aur = data['Average User Rating'].value_counts().sort_index()
#print(aur)
fig, ax = plt.subplots(1,2,figsize=(16, 4))
sns.countplot(data['Average User Rating'],ax=ax[0]) # seaborn
ax[1].bar(aur.index, aur, width=0.4) # matplotlib
ax[1].set_title('Average User Rating')
plt.show()"""
# reindex the data from 0-n
data.reset_index(drop=True, inplace=True)

# change date to datetime type
data_gap = np.zeros(n)
data_update = np.zeros(n)
data['Release_Date'] = pd.to_datetime(data['Original Release Date'],format='%d/%m/%Y')
data['Current_Date'] = pd.to_datetime(data['Current Version Release Date'],format='%d/%m/%Y')
#New column for time gap between release & update
data_gap=data.Current_Date-data.Release_Date
for i in range(n):
    data_update[i] = int(data_gap[i].days)
data['Update_Gap'] = data_update
#print(data['Update_Gap'])
data = data.drop(['Release_Date','Current_Date','Original Release Date','Current Version Release Date'],axis=1)

#print(data['Release_Date'],data['Current_Date'],data['Update_Gap'])

# the first column ['Name'], change it to 0 or 1
data_name = data['Name']
data_subtitle = data['Subtitle']
data_languages = data['Languages']
data = data.drop(['Name','Subtitle'],axis=1)

#print(n,data_name[7560])
#if('')
#print(range(n))

for i in range(n):
    if('free' not in data_name[i] and 'war' not in data_name[i] and 'game' not in data_name[i]):
        data_name[i]=0
    else:
        data_name[i]=1
data['Name'] = data_name
#print(data_name.value_counts())

# the same as ['Name'], I change subtitle to 0 or 1
for i in range(n):
   if(pd.isnull(data_subtitle[i])==0):
       if('strategy' not in data_subtitle[i] and 'puzzle' not in data_subtitle[i] and 'game' not in data_subtitle[i] and 'battle' not in data_subtitle[i]):
           data_subtitle[i]=0
       else:
           data_subtitle[i]=1
   else:
       data_subtitle[i] = 0
#print(data_subtitle.value_counts())
data['Subtitle'] = data_subtitle

# age, delete '+'


data['age_rating'] = data['Age Rating'].apply(lambda s : s.replace('+',''))
data = data.drop(['Age Rating'],axis=1)
NAR = [data.age_rating[(data['age_rating']=='4')].count(),data.age_rating[(data['age_rating']=='9')].count(),\
     data.age_rating[(data['age_rating']=='12')].count(),data.age_rating[(data['age_rating']=='17')].count()]
AR = ['Age 4+','Age 9+','Age 12+','Age 17+']

plt.pie(NAR, labels=AR, startangle=90, autopct='%.1f%%')
#plt.show()
#print(data['age_rating'])

#languages
for i in range(n):
    if(pd.isnull(data_languages[i])):
       data_languages[i] = 1
    else:
        data_languages[i] = (len(data_languages[i])+2)/4
data['Languages'] = data_languages

#print(data.describe(include='O').columns)


# replace 'games' and 'entertainment', other genres
data['GenreList'] = data['Genres'].apply(lambda s : s.replace('Games','').replace('&',' ').replace(',', ' ').replace('Entertainment','').replace('Strategy',''))
data = data.drop(['Genres'],axis=1)
#print(data['GenreList'].head())
data_genre = data['GenreList']
genre0 = np.zeros(n)
genre1 = np.zeros(n)
genre2 = np.zeros(n)
genre3 = np.zeros(n)
for i in range(n):
    if('Puzzle' in data_genre[i]):
        genre0[i] = 1
    else:
        genre0[i]=0
    if ('Simulation' in data_genre[i]):
        genre1[i] = 1
    else:
        genre1[i] = 0
    if ('Action' in data_genre[i]):
        genre2[i] = 1
    else:
        genre2[i] = 0
    if ('Board' in data_genre[i]):
        genre3[i] = 1
    else:
        genre3[i] = 0

data['Puzzle'] = genre0
data['Simulation'] = genre1
data['Action'] = genre2
data['Board'] = genre3
data = data.drop(['GenreList'],axis=1)

# in-app purchases
data_purchases = data['In-app Purchases']
data = data.drop(['In-app Purchases'],axis=1)
for i in range(n):
    if(pd.isnull(data_purchases[i])):
       data_purchases[i] = 0
    else:
       data_purchases[i] = 1
data['In-app Purchases'] = data_purchases

# merge ratings
new_ratings = []
for i, row in data.iterrows():
    rating = row['Average User Rating']
    if rating <= 4.0:
        ratings = 0
    else:
        ratings = 1
    new_ratings.append(ratings)
data['Average User Rating'] = new_ratings


#data.to_csv('E:\phd tools\ML_Problem2\data.csv')
data.to_csv(outputfile, index=False)

