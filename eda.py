import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, probplot

# set visualization style
plt.rcdefaults()
sns.set(style="ticks", color_codes=True, font_scale=1.5)
color = sns.color_palette()
sns.set_style('darkgrid')

def perform_basic_eda(user_song_list_count, count_play_df, track_metadata_df):
    """perform basic exploratory data analysis"""
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*80 + "\n")
    
    # basic statistics
    print(user_song_list_count.listen_count.describe().reset_index().T)
    
    # analyze heavy listeners
    print('\n{:d} users, {:.2%} of total play counts, listening a single song more than 200 times'.format(
        count_play_df.user[count_play_df.play_count>200].unique().shape[0],
        count_play_df.play_count[count_play_df.play_count>200].count()/count_play_df.shape[0]))
    print(count_play_df.play_count[count_play_df.play_count>200].describe().reset_index().T)
    
    # most obsessed user
    lunatic = count_play_df.play_count[count_play_df.play_count>count_play_df.play_count.max()-10].values[0]
    print('\n- How much days the most obsessed user can be the fan of a unique song: {:.1f}'.format((lunatic*3.5)/60/24))
    print(track_metadata_df[track_metadata_df.song_id.isin(
                      count_play_df[count_play_df.play_count>count_play_df.play_count.max()-10].song.values)])
    
    # second most obsessed
    obsessed = count_play_df.play_count[count_play_df.play_count>count_play_df.play_count.max()-1500].values[1]
    print('\n- How much days the second obsessed user can be the fan of a unique song: {:.1f}'.format((obsessed*3)/60/24))
    print(track_metadata_df[track_metadata_df.song_id.isin(count_play_df[count_play_df.play_count==920].song.values)])

def plot_top_popular_songs(user_song_list_count):
    """visualization 1: top 20 most popular songs"""
    fig = plt.figure(figsize=(13, 10))
    popular_songs = user_song_list_count[['title','listen_count']].groupby('title').sum().\
                    sort_values('listen_count', ascending=False).head(20).sort_values('listen_count')
    ax = popular_songs.plot(kind='barh', title='Top 20 Most Popular Songs', legend=False, color='steelblue')
    plt.xlabel('Listen Count')
    plt.ylabel('Song Title')
    for i, v in enumerate(popular_songs['listen_count']):
        ax.text(v + 50, i, str(int(v)), va='center')
    plt.tight_layout()
    plt.show()

def plot_top_popular_releases(user_song_list_count):
    """visualization 2: top 20 most popular releases"""
    fig = plt.figure(figsize=(13, 10))
    popular_release = user_song_list_count[['release','listen_count']].groupby('release').sum().\
                    sort_values('listen_count', ascending=False).head(20).sort_values('listen_count')
    ax = popular_release.plot(kind='barh', title='Top 20 Most Popular Releases', legend=False, color='coral')
    plt.xlabel('Listen Count')
    plt.ylabel('Release')
    for i, v in enumerate(popular_release['listen_count']):
        ax.text(v + 50, i, str(int(v)), va='center')
    plt.tight_layout()
    plt.show()

def plot_top_popular_artists(user_song_list_count):
    """visualization 3: top 20 most popular artists"""
    fig = plt.figure(figsize=(13, 10))
    popular_artist = user_song_list_count[['artist_name','listen_count']].groupby('artist_name').sum().\
                    sort_values('listen_count', ascending=False).head(20).sort_values('listen_count')
    ax = popular_artist.plot(kind='barh', title='Top 20 Most Popular Artists', legend=False, color='green')
    plt.xlabel('Listen Count')
    plt.ylabel('Artist')
    for i, v in enumerate(popular_artist['listen_count']):
        ax.text(v + 100, i, str(int(v)), va='center')
    plt.tight_layout()
    plt.show()

def plot_top_popular_years(user_song_list_count):
    """visualization 4: top 20 most popular years"""
    fig = plt.figure(figsize=(13, 10))
    popular_year = user_song_list_count[['year','listen_count']].groupby('year').sum().\
                    sort_values('listen_count', ascending=False).head(20).sort_values('listen_count')
    ax = popular_year.plot(kind='barh', title='Top 20 Most Popular Years of Song Listen', legend=False, color='purple')
    plt.xlabel('Listen Count')
    plt.ylabel('Year')
    for i, v in enumerate(popular_year['listen_count']):
        ax.text(v + 100, i, str(int(v)), va='center')
    plt.tight_layout()
    plt.show()

def qq_plot(data, measure):
    """create qq plot for distribution analysis"""
    fig = plt.figure(figsize=(20,7))

    (mu, sigma) = norm.fit(data)

    fig1 = fig.add_subplot(121)
    sns.distplot(data, fit=norm)
    fig1.set_title(measure + ' Distribution ( mu = {:.2f} and sigma = {:.2f} )'.format(mu, sigma), loc='center')
    fig1.set_xlabel(measure)
    fig1.set_ylabel('Frequency')

    fig2 = fig.add_subplot(122)
    res = probplot(data, plot=fig2)
    fig2.set_title(measure + ' Probability Plot (skewness: {:.6f} and kurtosis: {:.6f} )'.format(data.skew(), data.kurt()), loc='center')

    plt.tight_layout()
    plt.show()

def plot_user_song_distribution(user_song_list_count):
    """plot user song count distribution with qq plots"""
    user_song_count_distribution = user_song_list_count[['user','title']].groupby('user').count().\
                                    reset_index().sort_values(by='title', ascending=False)
    
    qq_plot(user_song_count_distribution.title, 'Song Title by User')
    print(user_song_count_distribution.title.describe().reset_index().T)
    
    qq_plot(user_song_list_count[user_song_list_count.title=="You're The One"].listen_count, 
            'Listen Most Popular Song')

def plot_listen_count_distribution(user_song_list_count):
    """visualization 5: distribution of listen counts"""
    fig = plt.figure(figsize=(12, 6))
    plt.hist(user_song_list_count.listen_count, bins=50, edgecolor='black', color='skyblue')
    plt.xlabel('Listen Count')
    plt.ylabel('Frequency')
    plt.title('Distribution of Listen Counts')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

def plot_user_activity_distribution(user_song_list_count):
    """visualization 6: user activity distribution"""
    fig = plt.figure(figsize=(12, 6))
    user_activity = user_song_list_count.groupby('user').size()
    plt.hist(user_activity, bins=50, edgecolor='black', color='lightcoral')
    plt.xlabel('Number of Songs per User')
    plt.ylabel('Number of Users')
    plt.title('User Activity Distribution')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

def plot_song_popularity_distribution(user_song_list_count):
    """visualization 7: song popularity distribution"""
    fig = plt.figure(figsize=(12, 6))
    song_popularity = user_song_list_count.groupby('song').size()
    plt.hist(song_popularity, bins=50, edgecolor='black', color='lightgreen')
    plt.xlabel('Number of Users per Song')
    plt.ylabel('Number of Songs')
    plt.title('Song Popularity Distribution')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

def plot_artist_diversity(user_song_list_count):
    """visualization 8: artist diversity"""
    fig = plt.figure(figsize=(12, 6))
    artist_songs = user_song_list_count.groupby('artist_name')['title'].nunique().sort_values(ascending=False).head(20)
    artist_songs.plot(kind='bar', color='orange', edgecolor='black')
    plt.xlabel('Artist')
    plt.ylabel('Number of Unique Songs')
    plt.title('Top 20 Artists by Number of Unique Songs')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_yearwise_song_count(user_song_list_count):
    """visualization 9: year-wise song count"""
    fig = plt.figure(figsize=(14, 6))
    year_count = user_song_list_count[user_song_list_count.year > 0].groupby('year').size()
    year_count.plot(kind='line', marker='o', color='navy', linewidth=2)
    plt.xlabel('Year')
    plt.ylabel('Number of Listens')
    plt.title('Listens by Year of Release')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_listen_count_heatmap(user_song_list_count):
    """visualization 10: listen count heatmap by top artists and years"""
    fig = plt.figure(figsize=(14, 8))
    top_artists = user_song_list_count.groupby('artist_name')['listen_count'].sum().sort_values(ascending=False).head(15).index
    filtered_data = user_song_list_count[(user_song_list_count.artist_name.isin(top_artists)) & 
                                         (user_song_list_count.year > 1990) & 
                                         (user_song_list_count.year < 2020)]
    pivot_data = filtered_data.pivot_table(values='listen_count', index='artist_name', 
                                           columns='year', aggfunc='sum', fill_value=0)
    sns.heatmap(pivot_data, cmap='YlOrRd', linewidths=0.5, cbar_kws={'label': 'Listen Count'})
    plt.title('Listen Count Heatmap: Top Artists by Year')
    plt.xlabel('Year')
    plt.ylabel('Artist')
    plt.tight_layout()
    plt.show()

def run_all_eda(user_song_list_count, count_play_df, track_metadata_df):
    """run all exploratory data analysis"""
    perform_basic_eda(user_song_list_count, count_play_df, track_metadata_df)
    
    print("\n--- Generating EDA Visualizations ---\n")
    
    plot_top_popular_songs(user_song_list_count)
    plot_top_popular_releases(user_song_list_count)
    plot_top_popular_artists(user_song_list_count)
    plot_top_popular_years(user_song_list_count)
    plot_user_song_distribution(user_song_list_count)
    plot_listen_count_distribution(user_song_list_count)
    plot_user_activity_distribution(user_song_list_count)
    plot_song_popularity_distribution(user_song_list_count)
    plot_artist_diversity(user_song_list_count)
    plot_yearwise_song_count(user_song_list_count)
    plot_listen_count_heatmap(user_song_list_count)
    
    print("\n--- EDA Complete: 10 Visualizations Generated ---\n")
