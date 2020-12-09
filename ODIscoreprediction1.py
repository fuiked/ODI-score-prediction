#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:50:28 2020

@author: looser
"""

import pandas as pd
df = pd.read_csv('/home/looser/Documents/matches.csv')
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler']
df.drop(labels=columns_to_remove, axis=1, inplace=True)
consistent_teams = ['England', 'Pakistan','Sri Lanka', 'Australia', 'South Africa',
       'New Zealand', 'Bangladesh', 'West Indies', 'India', 'Zimbabwe','Ireland']
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])
encoded_df = encoded_df[['date', 'bat_team_Australia', 'bat_team_Bangladesh',
       'bat_team_England', 'bat_team_India', 'bat_team_Ireland',
       'bat_team_New Zealand', 'bat_team_Pakistan', 'bat_team_South Africa',
       'bat_team_Sri Lanka', 'bat_team_West Indies', 'bat_team_Zimbabwe',
       'bowl_team_Australia', 'bowl_team_Bangladesh', 'bowl_team_England',
       'bowl_team_India', 'bowl_team_Ireland', 'bowl_team_New Zealand',
       'bowl_team_Pakistan', 'bowl_team_South Africa', 'bowl_team_Sri Lanka',
       'bowl_team_West Indies', 'bowl_team_Zimbabwe','overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'striker', 'non-striker', 'total']]
# Splitting the data into train and test set
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2006]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2007]
y_train = encoded_df[encoded_df['date'].dt.year <= 2006]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2007]['total'].values
# Removing the 'date' column
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)
print(X_test)
print(y_test)
from sklearn.ensemble import RandomForestRegressor
lin = RandomForestRegressor(n_estimators=100,max_features=None)
lin.fit(X_train,y_train)
lin.score(X_test, y_test)

team = ['England', 'Pakistan','Sri Lanka', 'Australia', 'South Africa',
       'New Zealand', 'Bangladesh', 'West Indies', 'India', 'Zimbabwe','Ireland']
print (team)
temp_array = list()
batting_team=input("enter batting team").capitalize()
if batting_team == 'Australia':
        temp_array = temp_array + [1,0,0,0,0,0,0,0,0,0,0]
elif batting_team == 'Bangladesh':
        temp_array = temp_array + [0,1,0,0,0,0,0,0,0,0,0]
elif batting_team == 'England':
        temp_array = temp_array + [0,0,1,0,0,0,0,0,0,0,0]
elif batting_team == 'India':
        temp_array = temp_array + [0,0,0,1,0,0,0,0,0,0,0]
elif batting_team == 'Ireland':
        temp_array = temp_array + [0,0,0,0,1,0,0,0,0,0,0]
elif batting_team == 'New zealand':
        temp_array = temp_array + [0,0,0,0,0,1,0,0,0,0,0]
elif batting_team == 'Pakistan':
        temp_array = temp_array + [0,0,0,0,0,0,1,0,0,0,0]
elif batting_team == 'South africa':
        temp_array = temp_array + [0,0,0,0,0,0,0,1,0,0,0]
elif batting_team == 'Sri lanka':
        temp_array = temp_array + [0,0,0,0,0,0,0,0,1,0,0]
elif batting_team == 'West indies':
        temp_array = temp_array + [0,0,0,0,0,0,0,0,0,1,0]
elif batting_team == 'Zimbabwe':
        temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,1]
else :
        print("enter valid team")
bowling_team=input("enter bowling team").capitalize()
if batting_team==bowling_team :
    print("you have selected same team in batting and bolwing")
else :
    if bowling_team == 'Australia':
        temp_array = temp_array + [1,0,0,0,0,0,0,0,0,0,0]
    elif bowling_team == 'Bangladesh':
        temp_array = temp_array + [0,1,0,0,0,0,0,0,0,0,0]
    elif bowling_team == 'England':
        temp_array = temp_array + [0,0,1,0,0,0,0,0,0,0,0]
    elif bowling_team == 'India':
        temp_array = temp_array + [0,0,0,1,0,0,0,0,0,0,0]
    elif bowling_team == 'Ireland':
        temp_array = temp_array + [0,0,0,0,1,0,0,0,0,0,0]
    elif bowling_team == 'New zealand':
        temp_array = temp_array + [0,0,0,0,0,1,0,0,0,0,0]
    elif bowling_team == 'Pakistan':
        temp_array = temp_array + [0,0,0,0,0,0,1,0,0,0,0]
    elif bowling_team == 'South africa':
        temp_array = temp_array + [0,0,0,0,0,0,0,1,0,0,0]
    elif bowling_team == 'Sri lanka':
        temp_array = temp_array + [0,0,0,0,0,0,0,0,1,0,0]
    elif bowling_team == 'West indies':
        temp_array = temp_array + [0,0,0,0,0,0,0,0,0,1,0]
    elif bowling_team == 'Zimbabwe':
        temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,1]
    else :
        print("enter valid team")
    overs = float(input('overs'))
    runs = int(input('runs'))
    wickets = int(input('wickets'))
    runs_in_prev_5 = int(input('runs_in_prev_5'))
    wickets_in_prev_5 = int(input('wickets_in_prev_5'))
    striker=int(input('striker runs'))
    nonstriker=int(input('non-striker runs'))
    import numpy as np
    temp_array = temp_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5, striker, nonstriker]
    data = np.array([temp_array])
    my_prediction = int(lin.predict(data)[0])
    print(my_prediction)
    