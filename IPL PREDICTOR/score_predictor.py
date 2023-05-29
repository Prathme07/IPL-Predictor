import pandas as pd
import numpy as np
df = pd.read_csv('best.csv')
#print(df.columns)
#print(df.shape)
#print(df.info())
#print(df.head())
#removing unwanted columns
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
#print('Before removing unwanted columns: {}'.format(df.shape))
df.drop(labels=columns_to_remove, axis=1, inplace=True)
#print('After removing unwanted columns: {}'.format(df.shape))
#print(df.head())
#print(df.index)
#print(df['bat_team'].unique())
df['bat_team'] = df['bat_team'].str.replace('Delhi Daredevils','Delhi Capitals')
df['bowl_team'] = df['bowl_team'].str.replace('Delhi Daredevils','Delhi Capitals')
#print(df['bat_team'].unique())
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Capitals', 'Sunrisers Hyderabad']
#print('Before removing inconsistent teams: {}'.format(df.shape))
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
#print('After removing inconsistent teams: {}'.format(df.shape))
df['bat_team'].unique()
# Removing the first 5 overs data in every match
#print('Before removing first 5 overs data: {}'.format(df.shape))
df = df[df['overs']>=5.0]
#print('After removing first 5 overs data: {}'.format(df.shape))
from datetime import datetime
#print("Before converting 'date' column from string to datetime object: {}".format(type(df.iloc[0,0])))
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
#print("After converting 'date' column from string to datetime object: {}".format(type(df.iloc[0,0])))
# Selecting correlated features using Heatmap
import matplotlib.pyplot as plt
import seaborn as sns

# Get correlation of all the features of the dataset
corr_matrix = df.corr()
top_corr_features = corr_matrix.index

# Plotting the heatmap
plt.figure(figsize=(13,10))
g = sns.heatmap(data=df[top_corr_features].corr(), annot=True, cmap='RdYlGn')
# Converting categorical features using OneHotEncoding method
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])
#print(encoded_df.columns)
# Rearranging the columns
encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Capitals', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Capitals', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]
# Splitting the data into train and test set
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2012]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2013]

y_train = encoded_df[encoded_df['date'].dt.year <= 2012]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2013]['total'].values

# Removing the 'date' column
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)

#print("Training set: {} and Test set: {}".format(X_train.shape, X_test.shape))
# Linear Regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred_linear = lr.predict(X_test)
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, accuracy_score
#print("---- Linear Regression - Model Evaluation ----")
#print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_linear)))
#print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_linear)))
#print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred_linear))))
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train,y_train)
y_pred_dt = dt.predict(X_test)
#print("---- Random Forest Regression - Model Evaluation ----")
#print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_dt)))
#print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_dt)))
#print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred_dt))))

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
y_pred_rfr = rfr.predict(X_test)
#print("---- Random Forest Regression - Model Evaluation ----")
#print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_rfr)))
#print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_rfr)))
#print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred_rfr))))
from sklearn.ensemble import AdaBoostRegressor
adb_regressor = AdaBoostRegressor(base_estimator=lr, n_estimators=100)
adb_regressor.fit(X_train, y_train)

# Predicting results
y_pred_adb = adb_regressor.predict(X_test)
#print("---- AdaBoost Regression - Model Evaluation ----")
#print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_adb)))
#print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_adb)))
#print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred_adb))))
def predict_score(batting_team='Chennai Super Kings', bowling_team='Mumbai Indians', overs=5.1, runs=50, wickets=0, runs_in_prev_5=50, wickets_in_prev_5=0):
  temp_array = list()

  # Batting Team
  if batting_team == 'Chennai Super Kings':
    temp_array = temp_array + [1,0,0,0,0,0,0,0]
  elif batting_team == 'Delhi Capitals':
    temp_array = temp_array + [0,1,0,0,0,0,0,0]
  elif batting_team == 'Kings XI Punjab':
    temp_array = temp_array + [0,0,1,0,0,0,0,0]
  elif batting_team == 'Kolkata Knight Riders':
    temp_array = temp_array + [0,0,0,1,0,0,0,0]
  elif batting_team == 'Mumbai Indians':
    temp_array = temp_array + [0,0,0,0,1,0,0,0]
  elif batting_team == 'Rajasthan Royals':
    temp_array = temp_array + [0,0,0,0,0,1,0,0]
  elif batting_team == 'Royal Challengers Bangalore':
    temp_array = temp_array + [0,0,0,0,0,0,1,0]
  elif batting_team == 'Sunrisers Hyderabad':
    temp_array = temp_array + [0,0,0,0,0,0,0,1]

  # Bowling Team
  if bowling_team == 'Chennai Super Kings':
    temp_array = temp_array + [1,0,0,0,0,0,0,0]
  elif bowling_team == 'Delhi Capitals':
    temp_array = temp_array + [0,1,0,0,0,0,0,0]
  elif bowling_team == 'Kings XI Punjab':
    temp_array = temp_array + [0,0,1,0,0,0,0,0]
  elif bowling_team == 'Kolkata Knight Riders':
    temp_array = temp_array + [0,0,0,1,0,0,0,0]
  elif bowling_team == 'Mumbai Indians':
    temp_array = temp_array + [0,0,0,0,1,0,0,0]
  elif bowling_team == 'Rajasthan Royals':
    temp_array = temp_array + [0,0,0,0,0,1,0,0]
  elif bowling_team == 'Royal Challengers Bangalore':
    temp_array = temp_array + [0,0,0,0,0,0,1,0]
  elif bowling_team == 'Sunrisers Hyderabad':
    temp_array = temp_array + [0,0,0,0,0,0,0,1]

  # Overs, Runs, Wickets, Runs_in_prev_5, Wickets_in_prev_5
  temp_array = temp_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]

  # Converting into numpy array
  temp_array = np.array([temp_array])

  # Prediction
  return int(lr.predict(temp_array)[0])


#PREDICTION 1
final_score = predict_score(batting_team='Royal Challengers Bangalore', bowling_team='Mumbai Indians', overs=8.4, runs=89, wickets=2, runs_in_prev_5=70, wickets_in_prev_5=1)
print("The final predicted score (range): {} to {}".format(final_score-10, final_score+5))