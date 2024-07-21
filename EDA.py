# Databricks notebook source
# MAGIC %md
# MAGIC Import Libraries

# COMMAND ----------

import pandas as pd
import numpy as np
# featuretools for automated feature engineering
# import featuretools as ft

# matplotlit and seaborn for visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 22
import seaborn as sns

# Suppress warnings from pandas
import warnings
warnings.filterwarnings('ignore')

# modeling 
import lightgbm as lgb

# utilities
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# memory management
import gc

# COMMAND ----------

# MAGIC %md
# MAGIC Import Data

# COMMAND ----------

players = pd.read_csv('/dbfs/FileStore/Group17_challenge/male_players.csv')
teams = pd.read_csv('/dbfs/FileStore/Group17_challenge/male_teams.csv')

# results = pd.read_csv('/dbfs/FileStore/Group17_challenge/2023_matchday_results.csv')
home = pd.read_csv('/dbfs/FileStore/Group17_challenge/2023_home_teams_stats.csv')
away = pd.read_csv('/dbfs/FileStore/Group17_challenge/2023_away_teams_stats.csv')
table = pd.read_csv('/dbfs/FileStore/Group17_challenge/2023_PL_standings.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC Choose important data

# COMMAND ----------

players = players[players['fifa_version']==23.0]
players = players[['player_id','club_name','short_name','player_positions','overall','potential','value_eur','wage_eur','age','height_cm','weight_kg','club_team_id','league_id','league_name','weak_foot','skill_moves','work_rate','defending_sliding_tackle','player_traits','pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic','attacking_crossing', 'attacking_finishing','attacking_heading_accuracy', 'attacking_short_passing','attacking_volleys', 'skill_dribbling', 'skill_curve',
'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
'movement_reactions', 'movement_balance', 'power_shot_power','power_jumping', 'power_stamina', 'power_strength', 'power_long_shots','mentality_aggression', 'mentality_interceptions','mentality_positioning', 'mentality_vision', 'mentality_penalties','mentality_composure', 'defending_marking_awareness','defending_standing_tackle', 'defending_sliding_tackle']]

teams = teams[(teams['league_id']==13) & (teams['fifa_version']==23)]
teams = teams[['league_name','team_name','def_team_width','def_team_depth','def_defence_pressure','def_defence_aggression','def_defence_width','def_defence_defender_line','off_build_up_play','off_chance_creation','off_team_width','off_players_in_box','off_corners','off_free_kicks','build_up_play_speed','build_up_play_dribbling','build_up_play_passing','build_up_play_positioning','chance_creation_passing','chance_creation_crossing','chance_creation_shooting','chance_creation_positioning']]
teams = teams.drop(columns = ['def_defence_pressure','def_defence_aggression','def_defence_width','def_defence_defender_line','build_up_play_speed','build_up_play_dribbling','build_up_play_positioning','chance_creation_passing','chance_creation_crossing','chance_creation_shooting','chance_creation_positioning','build_up_play_passing'])

# check_nans
# players[players['value_eur'].isna()].overall.value_counts()
# so it looks like value_eur is nan where they are basically still worthless
# players[players['club_team_id'].isna()] 
# so it looks like club_team_id is nan where they are not in any club, we don't them for modeling, maybe later
# remove player_traits and mentality_composure
# players[players['pace'].isna()].player_positions.unique()
# looks like those 20024 are goalkeepers, for now just drop it
players = players.drop(columns = ['player_traits','mentality_composure','player_id'])
players_clean = players.dropna() # bo wiemy ze reszta sie nie przyda
players_clean = players_clean[(players_clean['league_name']=='Premier League') | (players_clean['league_name']=='Championship')]
# okay lets say its clean now
# lets stick to premier league now and 22/23 season now
# okay teams cleaned

# COMMAND ----------

# MAGIC %md
# MAGIC REMEMBER
# MAGIC - add league lvl from ranking

# COMMAND ----------

# results_home = results[['teams.home.name','goals.home','goals.away']]
# results_away = results[['teams.away.name','goals.home','goals.away']]

# results_home['goals_scored'] = results_home.groupby('teams.home.name')['goals.home'].transform('sum')
# results_home['goals_conceeded'] = results_home.groupby('teams.home.name')['goals.away'].transform('sum')
# results_home = results_home.rename(columns={'teams.home.name':'team_name'})[['team_name','goals_scored','goals_conceeded']]

# results_away['goals_scored'] = results_away.groupby('teams.away.name')['goals.away'].transform('sum')
# results_away['goals_conceeded'] = results_away.groupby('teams.away.name')['goals.home'].transform('sum')
# results_away = results_away.rename(columns={'teams.away.name':'team_name'})[['team_name','goals_scored','goals_conceeded']]

# full = pd.concat([results_away.drop_duplicates(),results_home.drop_duplicates()])
# full['goals_scored_full'] = full.groupby('team_name')['goals_scored'].transform('sum')
# full['goals_conceeded_full'] = full.groupby('team_name')['goals_conceeded'].transform('sum')
# full = full[['team_name','goals_scored_full','goals_conceeded_full']].drop_duplicates()

# #nevermind I've found it in other table

# COMMAND ----------

table = table[['team.name','points','goals_for','goals_against']]

full = pd.concat([home.drop(columns=['fixture id','Home team id','Yellow Cards','Red Cards','Offsides']).rename(columns={'Home team name':'team_name'}),
           away.drop(columns=['fixture id','away team id','Yellow Cards','Red Cards','Offsides']).rename(columns={'away team name':'team_name'})])

full['Shots on Goal'] = full['Shots on Goal'].fillna(full.groupby('team_name')['Shots on Goal'].transform('mean'))
full['Shots off Goal'] = full['Shots off Goal'].fillna(full.groupby('team_name')['Shots off Goal'].transform('mean'))
full['Blocked Shots'] = full['Blocked Shots'].fillna(full.groupby('team_name')['Blocked Shots'].transform('mean'))
full['Shots insidebox'] = full['Shots insidebox'].fillna(full.groupby('team_name')['Shots insidebox'].transform('mean'))
full['Shots outsidebox'] = full['Shots outsidebox'].fillna(full.groupby('team_name')['Shots outsidebox'].transform('mean'))
full['Corner Kicks'] = full['Corner Kicks'].fillna(full.groupby('team_name')['Corner Kicks'].transform('mean'))
full['Goalkeeper Saves'] = full['Goalkeeper Saves'].fillna(full.groupby('team_name')['Goalkeeper Saves'].transform('mean'))
full['expected_goals'] = full['expected_goals'].fillna(full.groupby('team_name')['expected_goals'].transform('mean'))

full['Ball Possession'] = full['Ball Possession'].apply(lambda x: float(x.split('%')[0])*0.01)
full['Passes %'] = full['Passes %'].apply(lambda x: float(x.split('%')[0])*0.01)

for column in full.columns:
    if column !='team_name':
        full[column] = full.groupby('team_name')[column].transform('mean')
full = full.drop_duplicates()

table_with_results = table.rename(columns={'team.name':'team_name'}).merge(full)

# COMMAND ----------

players_clean.loc[players_clean['club_name']=='Tottenham Hotspur', 'club_name'] = 'Tottenham'
players_clean.loc[players_clean['club_name']=='Newcastle United', 'club_name'] = 'Newcastle'
players_clean.loc[players_clean['club_name']=='West Ham United', 'club_name'] = 'West Ham'
players_clean.loc[players_clean['club_name']=='Brighton & Hove Albion', 'club_name'] = 'Brighton'
players_clean.loc[players_clean['club_name']=='Leicester City', 'club_name'] = 'Leicester'
players_clean.loc[players_clean['club_name']=='Wolverhampton Wanderers', 'club_name'] = 'Wolves'
players_clean.loc[players_clean['club_name']=='AFC Bournemouth', 'club_name'] = 'Bournemouth'
players_clean.loc[players_clean['club_name']=='Leeds United', 'club_name'] = 'Leeds'
players_clean = players_clean.rename(columns = {'club_name':'team_name'})
players_clean = players_clean[players_clean['team_name'].isin(table_with_results['team_name'].unique())]

# COMMAND ----------

# MAGIC %md
# MAGIC So now we have two tables, players with stats from fifa and stats from 22/23 season premier league

# COMMAND ----------

table_with_results.head(5)

# COMMAND ----------

players_clean.head(5)

# COMMAND ----------

# okay lets see ow many players for club we ave
players_clean['players_count'] = players_clean.groupby('team_name')['short_name'].transform('count')
players_clean[['team_name','players_count']].drop_duplicates() # okey jest niezle, ale zobaczmy kto wgl rozegral wiecej meczow
# okay, jako ze nie mam niestety ilosci zagranych meczow przez zawodnikow, to wezmiemy po prostu top 10 graczy na pozycje
players_clean = players_clean.drop(columns = 'players_count')

# COMMAND ----------

players_clean['player_positions'] = players_clean['player_positions'].apply(lambda x: x.split(',')[-1])
players_clean['player_positions'] = players_clean['player_positions'].apply(lambda x: x.split(' ')[-1])
attacker=['RW', 'ST', 'LW','CF']
midfielder=['CM', 'CDM', 'LM', 'CAM','RM']
defender=['LB','LWB','CB','RWB','RB']
players_clean['player_positions'] =players_clean['player_positions'].apply(lambda x: 'attacker' if x in attacker else x)
players_clean['player_positions'] =players_clean['player_positions'].apply(lambda x: 'midfielder' if x in midfielder else x)
players_clean['player_positions'] =players_clean['player_positions'].apply(lambda x: 'defender' if x in defender else x)

# COMMAND ----------

players_clean.player_positions.value_counts() # lets say, ze ma to sens, jedziemy dalej

# COMMAND ----------

players__position_counts = players_clean.groupby(['team_name','player_positions'])['short_name'].agg('count').reset_index()
# players_counts = players_clean.groupby(['team_name'])['short_name'].agg('count').reset_index(name='team_quantity')# nie widze tu zadnej korelacji WYRZUCAM
players_counts_over70 = players_clean[players_clean['overall']>=70].groupby(['team_name','player_positions'])['short_name'].agg('count').reset_index()

# COMMAND ----------

import matplotlib.pyplot as plt
plt.boxplot(players_clean[players_clean['player_positions']=='attacker'].overall)
plt.show()
plt.boxplot(players_clean[players_clean['player_positions']=='midfielder'].overall)
plt.show()
plt.boxplot(players_clean[players_clean['player_positions']=='defender'].overall)
plt.show()

# COMMAND ----------

players__position_counts # no wyglada na to ze za duzo nie mozemy wyciac
players_counts_over70 # no wyglada na to ze za duzo nie mozemy wyciac

# COMMAND ----------

import plotly.express as px
ova_per_club = players_clean.groupby('team_name')['overall'].agg('mean').reset_index(name='mean_overall').sort_values(by='mean_overall',ascending=False)
fig=px.bar(ova_per_club,x='team_name',y='mean_overall',color='mean_overall',title='mean overall per club')
fig.show()

# COMMAND ----------

# lets see which skills are required on different positions
stats = list(players_clean.iloc[:,-35:].columns)
players_over = players_clean[players_clean['overall']>70]
pos_overall = players_over.groupby('player_positions')[stats].agg('mean').reset_index()
pos_overall = pos_overall.T.drop_duplicates()
pos_overall = pos_overall.T.drop_duplicates()

print('Overall Attributes of the Players in FIFA 23 over 70 overall')
fig=plt.figure(figsize=(30,30))

for i in range(0,3):

    total=pos_overall.loc[i,stats].values
    angles=np.linspace(0, 2*np.pi, len(stats), endpoint=False) 
    ax = fig.add_subplot(3,1,i+1, polar=True)
    ax.plot(angles, total, 'o-', linewidth=1)
    ax.fill(angles, total,color='red',alpha=0.25)
    ax.set_thetagrids(angles * 180/np.pi, stats)
    ax.set_title([pos_overall.loc[i,"player_positions"]])
    ax.grid(True)

# COMMAND ----------

# MAGIC %md
# MAGIC Okay so we have our final table with overall>70

# COMMAND ----------

table_results_with_player = table_with_results.merge(players_over.groupby(['team_name','player_positions'])[stats].agg('mean').reset_index(), how='left')

# COMMAND ----------

def_players_stats = table_results_with_player[table_results_with_player['player_positions'].isin(['midfielder','defender'])]
off_players_stats = table_results_with_player[table_results_with_player['player_positions'].isin(['attacker','midfielder'])]

# COMMAND ----------

numeric_columns = table_results_with_player.select_dtypes(include=["float",'int64']).columns.tolist()

# COMMAND ----------

table_results_with_player.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC POINTS SCORED BY WHOLE TEAM

# COMMAND ----------

import seaborn as sns
correlation = table_results_with_player[stats+['points']].corr()
plt.figure(figsize=(10,8), dpi =100)
sns.heatmap(correlation[['points']].sort_values(by='points',ascending=False).head(10), linewidth=.5)
plt.show()
correlation[['points']].sort_values(by='points',ascending=False).head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC POINTS SCORED BY OFFENSIVE PLAYERS

# COMMAND ----------

import seaborn as sns
correlation = off_players_stats[stats+['points']].corr()
plt.figure(figsize=(10,8), dpi =100)
sns.heatmap(correlation[['points']].sort_values(by='points',ascending=False).head(10), linewidth=.5)
plt.show()
correlation[['points']].sort_values(by='points',ascending=False).head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC THERE IS a difference so, we'll have a look only in terms of positions
# MAGIC GOALS

# COMMAND ----------

import seaborn as sns
correlation = off_players_stats[stats+['goals_for']].corr()
plt.figure(figsize=(10,8), dpi =100)
sns.heatmap(correlation[['goals_for']].sort_values(by='goals_for',ascending=False).head(10), linewidth=.5)
plt.show()
correlation[['goals_for']].sort_values(by='goals_for',ascending=False).head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC GOALS AGGAINST FOR DEFENSIVE PLAYERS # CHECK ASCENDING TRUE, interesinst stat

# COMMAND ----------

import seaborn as sns
correlation = def_players_stats[def_players_stats['player_positions']=='defender'][stats+['goals_against']].corr()
plt.figure(figsize=(10,8), dpi =100)
sns.heatmap(correlation[['goals_against']].sort_values(by='goals_against',ascending=True).head(10), linewidth=.5)
plt.show()
correlation[['goals_against']].sort_values(by='goals_against',ascending=True).head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC POINTS VS DEF PLAYERS

# COMMAND ----------

import seaborn as sns
correlation = def_players_stats[stats+['points']].corr()
plt.figure(figsize=(5,5), dpi =100)
sns.heatmap(correlation[['points']].sort_values(by='points',ascending=False).head(10), linewidth=.5)
plt.show()
correlation[['points']].sort_values(by='points',ascending=False).head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC POINTS VS whole team

# COMMAND ----------

import seaborn as sns
correlation = table_with_results.corr()
plt.figure(figsize=(5,5), dpi =100)
sns.heatmap(correlation[['points']].sort_values(by='points',ascending=False).head(20), linewidth=.5)
plt.show()
correlation[['points']].sort_values(by='points',ascending=False).head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC goals_scored VS whole team

# COMMAND ----------

import seaborn as sns
correlation = table_with_results.corr()
plt.figure(figsize=(5,5), dpi =100)
sns.heatmap(correlation[['goals_for']].sort_values(by='goals_for',ascending=False).head(20), linewidth=.5)
plt.show()
correlation[['goals_for']].sort_values(by='goals_for',ascending=False).head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC goals_aggainst VS whole team

# COMMAND ----------

import seaborn as sns
correlation = table_with_results.corr()
plt.figure(figsize=(5,5), dpi =100)
sns.heatmap(correlation[['goals_against']].sort_values(by='goals_against',ascending=True).head(20), linewidth=.5)
plt.show()
correlation[['goals_against']].sort_values(by='goals_against',ascending=True).head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC What do we want to find : 
# MAGIC 1. what has most influence on winning team in a legue (defensive stats vs offensive stats, we dont have much but it must be enough)
# MAGIC 2. find what players has what You need
# MAGIC 3. find those players but for chaper price

# COMMAND ----------

# MAGIC %md
# MAGIC FEATURE SELECTION FOR DEFENSIVE PLAYERS
# MAGIC musze wyrzucic 
# MAGIC , 'Shots on Goal', 'goals_for', 'goals_against' bo obviously sa za mocne

# COMMAND ----------

def drop_columns(df, delete_cols):
    for col in delete_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    return df

def delete_correlated_cols(df):
    train = drop_columns(df, ['team_name','player_positions', 'Shots on Goal', 'goals_for', 'goals_against'])

    train_labels = train["points"]
    # Threshold for removing correlated variables
    threshold = 0.9

    # Absolute value correlation matrix
    corr_matrix = train.drop(columns='points').corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Select columns with correlations above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print('There are %d columns to remove.' % (len(to_drop)))
    df = drop_columns(df, to_drop +  ['team_name','player_positions', 'Shots on Goal', 'goals_for', 'goals_against'])
    return df

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def feature_selection_importances(df):
    top_features_full = pd.DataFrame()
    for i in range(0,3):
        # Assuming df is your DataFrame and 'points' is the target column
        X = df.drop(columns=['points'])
        y = df['points']

        # Split the data into training and testing sets
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and fit the RandomForestRegressor
        model = RandomForestRegressor(random_state=42)
        # model.fit(X_train, y_train)
        model.fit(X, y)

        # Get feature importances
        feature_importances = model.feature_importances_
        features = X.columns

        # Create a DataFrame for better visualization
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        top_features_full = pd.concat([top_features_full, importance_df])
    top_feats = list(top_features_full.groupby('Feature').agg('mean').reset_index().sort_values('Importance').tail(7).Feature.unique())
    return df[['points'] + top_feats], top_features_full.groupby('Feature').agg('mean').reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC First approach with all the stats

# COMMAND ----------

# clean_def_stats = delete_correlated_cols(def_players_stats)
# clean_off_stats = delete_correlated_cols(off_players_stats)
# clean_def_stats, features_def = feature_selection_importances(clean_def_stats)
# clean_off_stats, features_off = feature_selection_importances(clean_off_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC Second approach with only specific player approach (so for now the easier one)

# COMMAND ----------

# players_clean.columns
def_players_stats = table_results_with_player[table_results_with_player['player_positions'].isin(['midfielder','defender'])]
off_players_stats = table_results_with_player[table_results_with_player['player_positions'].isin(['attacker','midfielder'])]
only_player_stats = [elem for elem in def_players_stats.columns if elem in players_clean.columns]

# COMMAND ----------

players_def_players_stats = def_players_stats[only_player_stats + ['points']]
players_off_players_stats = off_players_stats[only_player_stats + ['points']]
players_clean_def_stats = delete_correlated_cols(players_def_players_stats)
players_clean_off_stats = delete_correlated_cols(players_off_players_stats)
players_clean_def_stats, features_def = feature_selection_importances(players_clean_def_stats)
players_clean_off_stats, features_off = feature_selection_importances(players_clean_off_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC MODELING SECOND APPROACH

# COMMAND ----------

# MAGIC %md
# MAGIC UNDER AND OVER PERFORMING TEAMS (linear regression)

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_model(players_stats, model):
    # Define feature matrix (X) and target vector (y)
    X = players_stats.drop(columns=['points'])
    y = players_stats['points']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    return model

def predict_and_evaluate(model, players_stats, players_full_stats):
    # Group the final dataframes
    final_off = players_stats.drop(columns='player_positions').groupby('team_name').agg('mean').reset_index()

    # Find common columns
    common = [elem for elem in players_full_stats.columns if elem in final_off.columns]

    # Extract the relevant columns, including 'team_name' and 'points'
    player_stats = final_off[common + ['team_name', 'points']].copy()

    # Ensure the features in player_stats match those used in the model
    features = player_stats.drop(columns=['team_name', 'points'])

    # Predict points using the trained model
    player_points_pred = model.predict(features)

    # Add predicted points to the player_stats DataFrame
    player_stats['predicted_points'] = player_points_pred

    # clean double points column
    player_stats = player_stats.iloc[:,1:]

    # Calculate the difference
    player_stats['difference'] = player_stats.apply(lambda row: row['points'] - row['predicted_points'], axis=1).round(1)

    return player_stats

def visualisation_performance(df, title_suf):
    df = df.sort_values('difference')
    # Create a larger bar plot
    plt.figure(figsize=(14, 8))  # Adjusted figure size
    colors = np.where(df['difference'] >= 0, 'green', 'red')
    bars = plt.bar(df['team_name'], df['difference'], color=colors)

    # Add labels and title
    plt.xlabel('Team Name')
    plt.ylabel('Difference in Points')
    plt.title(f'Overperforming and Underperforming Teams {title_suf}')
    plt.axhline(0, color='black', linewidth=0.3)  # Add a horizontal line at y=0

    # Rotate x-axis labels for better fit
    plt.xticks(rotation=45, ha='right')

    # Annotate the bars
    for bar, diff in zip(bars, df['difference']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + (1 if height >= 0 else -3), f'{diff}', ha='center', va='bottom' if height >= 0 else 'top')

    # Show plot
    plt.tight_layout()  # Adjust layout to fit everything
    plt.show()

model = train_model(players_clean_off_stats, LinearRegression())
player_stats = predict_and_evaluate(model, off_players_stats, players_clean_off_stats)
visualisation_performance(player_stats, 'offensively')

model = train_model(players_clean_def_stats, LinearRegression())
player_stats = predict_and_evaluate(model, def_players_stats, players_clean_def_stats)
visualisation_performance(player_stats, 'defensively')

# COMMAND ----------

# MAGIC %md
# MAGIC random forest

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor

model = train_model(players_clean_off_stats, RandomForestRegressor(n_estimators=100, random_state=42))
player_stats = predict_and_evaluate(model, off_players_stats, players_clean_off_stats)
visualisation_performance(player_stats, 'offensively')

model = train_model(players_clean_def_stats, RandomForestRegressor(n_estimators=100, random_state=42))
player_stats = predict_and_evaluate(model, def_players_stats, players_clean_def_stats)
visualisation_performance(player_stats, 'defensively')

# COMMAND ----------

# MAGIC %md
# MAGIC okay lets back to the most important topic, finding perfect players

# COMMAND ----------

players_clean_off_stats
