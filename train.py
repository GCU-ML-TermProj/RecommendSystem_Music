import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from util import (popularity_recommender_py, item_similarity_recommender_py, 
                  compute_svd, compute_estimated_matrix, show_recomendations,
                  create_popularity_recommendation)

def train_popularity_model(user_song_list_count):
    """train popularity based recommendation model"""
    print("\n" + "="*80)
    print("POPULARITY-BASED RECOMMENDATION ENGINE")
    print("="*80 + "\n")
    
    recommendations = create_popularity_recommendation(user_song_list_count, 'user', 'title', 15)
    print("Top 15 popular songs:")
    print(recommendations)
    
    print("\nTop 10 popular artists:")
    artist_recommendations = create_popularity_recommendation(user_song_list_count, 'user', 'artist_name', 10)
    print(artist_recommendations)
    
    return recommendations

def train_item_similarity_model(user_song_list_count):
    """train item similarity based recommendation model"""
    print("\n" + "="*80)
    print("ITEM SIMILARITY BASED RECOMMENDATION ENGINE")
    print("="*80 + "\n")
    
    # select top 5000 songs representing majority of listens
    total_play_count = sum(user_song_list_count.listen_count)
    play_count = user_song_list_count[['song', 'listen_count']].groupby('song').sum().\
                 sort_values(by='listen_count', ascending=False).head(5000)
    print('5,000 most popular songs represents {:3.2%} of total listen.'.format(float(play_count.sum())/total_play_count))
    
    song_subset = list(play_count.index[:5000])
    user_subset = list(user_song_list_count.loc[user_song_list_count.song.isin(song_subset), 'user'].unique())
    user_song_list_count_sub = user_song_list_count[user_song_list_count.song.isin(song_subset)]
    print(user_song_list_count_sub.head())
    
    # create and train model
    is_model = item_similarity_recommender_py()
    is_model.create(user_song_list_count_sub, 'user', 'title')
    
    # test on a sample user
    user_id = list(user_song_list_count_sub.user)[7]
    user_items = is_model.get_user_items(user_id)
    recommendations = is_model.recommend(user_id)
    
    print(f"\nRecommendations for user {user_id}:")
    print(recommendations)
    
    print(f"\nUser's listening history (>5 plays):")
    print(user_song_list_count_sub[(user_song_list_count_sub.user==user_id) & 
                                   (user_song_list_count_sub.listen_count>5)])
    
    return is_model, user_song_list_count_sub

def train_matrix_factorization_model(user_song_list_count):
    """train matrix factorization based recommendation model using svd"""
    print("\n" + "="*80)
    print("MATRIX FACTORIZATION BASED RECOMMENDATIONS")
    print("="*80 + "\n")
    
    # create fractional play count
    print("Creating fractional play count...")
    user_song_list_listen = user_song_list_count[['user','listen_count']].groupby('user').sum().reset_index()
    user_song_list_listen.rename(columns={'listen_count':'total_listen_count'}, inplace=True)
    user_song_list_count_merged = pd.merge(user_song_list_count, user_song_list_listen)
    user_song_list_count_merged['fractional_play_count'] = \
        user_song_list_count_merged['listen_count']/user_song_list_count_merged['total_listen_count']
    
    sample_user = 'd6589314c0a9bcbca4fee0c93b14bc402363afea'
    print(f"\nSample user {sample_user} fractional play counts:")
    print(user_song_list_count_merged[user_song_list_count_merged.user == sample_user]\
          [['user','song','listen_count','fractional_play_count']].head())
    
    # create sparse matrix
    print("\nCreating sparse matrix...")
    user_codes = user_song_list_count_merged.user.drop_duplicates().reset_index()
    user_codes.rename(columns={'index':'user_index'}, inplace=True)
    user_codes['us_index_value'] = list(user_codes.index)
    
    song_codes = user_song_list_count_merged.song.drop_duplicates().reset_index()
    song_codes.rename(columns={'index':'song_index'}, inplace=True)
    song_codes['so_index_value'] = list(song_codes.index)
    
    small_set = pd.merge(user_song_list_count_merged, song_codes, how='left')
    small_set = pd.merge(small_set, user_codes, how='left')
    
    mat_candidate = small_set[['us_index_value','so_index_value','fractional_play_count']]
    data_array = mat_candidate.fractional_play_count.values
    row_array = mat_candidate.us_index_value.values
    col_array = mat_candidate.so_index_value.values
    
    data_sparse = coo_matrix((data_array, (row_array, col_array)), dtype=float)
    print(f"Sparse matrix shape: {data_sparse.shape}")
    print(data_sparse)
    
    # run svd
    print("\nRunning SVD...")
    K = 50
    urm = data_sparse
    MAX_PID = urm.shape[1]
    MAX_UID = urm.shape[0]
    U, S, Vt = compute_svd(urm, K)
    
    # test on sample users
    uTest = [4, 5, 6, 7, 8, 873, 23]
    uTest_recommended_items = compute_estimated_matrix(urm, U, S, Vt, uTest, K, True, MAX_UID, MAX_PID)
    
    print("\n--- Recommendations for Test Users ---")
    show_recomendations(uTest, uTest_recommended_items, small_set)
    
    # test on user 0
    uTest = [0]
    print("\n--- Recommendations for User 0 ---")
    uTest_recommended_items = compute_estimated_matrix(urm, U, S, Vt, uTest, K, True, MAX_UID, MAX_PID)
    show_recomendations(uTest, uTest_recommended_items, small_set)
    
    return U, S, Vt, small_set, urm

def train_surprise_models(user_song_list_count, unique_track_metadata_df, eval_df_merged):
    """train knn and svd++ models using surprise library"""
    from surprise import KNNWithMeans, SVDpp, Dataset, Reader
    from surprise import accuracy, Prediction
    
    print("\n" + "="*80)
    print("ENHANCED RECOMMENDATION SYSTEM WITH SURPRISE LIBRARY")
    print("="*80 + "\n")
    
    # prepare data: 70% train + 30% test split
    print("\n--- Preparing Data: 70% Train + 30% Test Split ---")
    
    if eval_df_merged is not None:
        # use kaggle evaluation file
        top_users_count = user_song_list_count.groupby('user').size().sort_values(ascending=False).head(1500)
        top_songs_count = user_song_list_count.groupby('song').size().sort_values(ascending=False).head(1500)
        
        # filter training data
        train_data_filtered = user_song_list_count[
            (user_song_list_count.user.isin(top_users_count.index)) & 
            (user_song_list_count.song.isin(top_songs_count.index))
        ].copy()
        
        # filter eval data to same users/songs
        eval_data_filtered = eval_df_merged[
            (eval_df_merged.user.isin(top_users_count.index)) & 
            (eval_df_merged.song.isin(top_songs_count.index))
        ].copy()
        
        print(f"Filtered train data: {train_data_filtered.shape}")
        print(f"Filtered eval data: {eval_data_filtered.shape}")
        
        # sample 30% from eval data
        if len(eval_data_filtered) > 0:
            eval_sample = eval_data_filtered.sample(frac=0.3, random_state=42)
        else:
            eval_sample = pd.DataFrame()
        
        # sample 70% from train data
        train_sample = train_data_filtered.sample(frac=0.7, random_state=42)
        
        print(f"\n30% from eval: {len(eval_sample)} samples")
        print(f"70% from train: {len(train_sample)} samples")
        
        # combine for new test set
        combined_test_data = pd.concat([eval_sample, train_sample], ignore_index=True)
        surprise_data = train_data_filtered.copy()
        
    else:
        # fallback: use only training data with 80-20 split
        print("Using only training data with 80-20 split")
        
        top_users_count = user_song_list_count.groupby('user').size().sort_values(ascending=False).head(1500)
        top_songs_count = user_song_list_count.groupby('song').size().sort_values(ascending=False).head(1500)
        
        surprise_data = user_song_list_count[
            (user_song_list_count.user.isin(top_users_count.index)) & 
            (user_song_list_count.song.isin(top_songs_count.index))
        ].copy()
        
        combined_test_data = surprise_data.sample(frac=0.2, random_state=42)
    
    print(f"\nFinal training data shape: {surprise_data.shape}")
    print(f"Combined test data: {combined_test_data.shape}")
    print(f"Number of unique users: {combined_test_data.user.nunique()}")
    print(f"Number of unique songs: {combined_test_data.song.nunique()}")
    print(f"Matrix sparsity: {1 - (len(surprise_data) / (surprise_data.user.nunique() * surprise_data.song.nunique())):.4f}")
    
    # normalize ratings
    surprise_data['rating'] = surprise_data['listen_count'].apply(
        lambda x: min(5, max(1, 1 + 4 * (np.log1p(x) / np.log1p(surprise_data['listen_count'].max()))))
    )
    
    combined_test_data['rating'] = combined_test_data['listen_count'].apply(
        lambda x: min(5, max(1, 1 + 4 * (np.log1p(x) / np.log1p(combined_test_data['listen_count'].max()))))
    )
    
    # create surprise datasets
    reader = Reader(rating_scale=(1, 5))
    surprise_dataset = Dataset.load_from_df(surprise_data[['user', 'song', 'rating']], reader)
    trainset = surprise_dataset.build_full_trainset()
    
    # create test set
    testset = [(row['user'], row['song'], row['rating']) 
               for _, row in combined_test_data.iterrows()]
    
    print(f"\nTrainset size: {trainset.n_ratings}")
    print(f"Testset size: {len(testset)}")
    
    # train knn item-based collaborative filtering
    print("\n--- Training Item-Based Collaborative Filtering (KNN) Model ---")
    sim_options = {'name': 'cosine', 'user_based': False}
    knn_item_model = KNNWithMeans(k=40, sim_options=sim_options, verbose=True)
    knn_item_model.fit(trainset)
    
    print("\n--- Making Predictions with KNN Item-Based Model ---")
    knn_predictions = knn_item_model.test(testset)
    knn_rmse = accuracy.rmse(knn_predictions, verbose=True)
    knn_mae = accuracy.mae(knn_predictions, verbose=True)
    
    # train svd++ model
    print("\n--- Training SVD++ Model ---")
    svdpp_model = SVDpp(n_factors=20, n_epochs=20, lr_all=0.007, reg_all=0.02, verbose=True)
    svdpp_model.fit(trainset)
    
    print("\n--- Making Predictions with SVD++ Model ---")
    svdpp_predictions = svdpp_model.test(testset)
    svdpp_rmse = accuracy.rmse(svdpp_predictions, verbose=True)
    svdpp_mae = accuracy.mae(svdpp_predictions, verbose=True)
    
    # get top-n recommendations
    def get_top_n_recommendations(predictions, n=10):
        top_n = {}
        for uid, iid, true_r, est, _ in predictions:
            if uid not in top_n:
                top_n[uid] = []
            top_n[uid].append((iid, est))
        
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]
        
        return top_n
    
    knn_top_n = get_top_n_recommendations(knn_predictions, n=10)
    svdpp_top_n = get_top_n_recommendations(svdpp_predictions, n=10)
    
    # display sample recommendations
    sample_user = list(knn_top_n.keys())[0]
    print(f"\n--- Top 10 Recommendations for User: {sample_user} ---")
    
    print("\nKNN Item-Based CF:")
    for i, (song, rating) in enumerate(knn_top_n[sample_user], 1):
        song_title = surprise_data[surprise_data.song == song]['title'].iloc[0] if len(surprise_data[surprise_data.song == song]) > 0 else "Unknown"
        print(f"{i}. {song_title[:50]} (Est. Rating: {rating:.2f})")
    
    print("\nSVD++:")
    for i, (song, rating) in enumerate(svdpp_top_n[sample_user], 1):
        song_title = surprise_data[surprise_data.song == song]['title'].iloc[0] if len(surprise_data[surprise_data.song == song]) > 0 else "Unknown"
        print(f"{i}. {song_title[:50]} (Est. Rating: {rating:.2f})")
    
    return {
        'knn_model': knn_item_model,
        'svdpp_model': svdpp_model,
        'knn_predictions': knn_predictions,
        'svdpp_predictions': svdpp_predictions,
        'knn_rmse': knn_rmse,
        'knn_mae': knn_mae,
        'svdpp_rmse': svdpp_rmse,
        'svdpp_mae': svdpp_mae,
        'surprise_data': surprise_data,
        'testset': testset
    }

def run_all_training(user_song_list_count, unique_track_metadata_df, skip_surprise=False):
    """run all model training"""
    # train basic models
    popularity_recs = train_popularity_model(user_song_list_count)
    is_model, user_song_sub = train_item_similarity_model(user_song_list_count)
    U, S, Vt, small_set, urm = train_matrix_factorization_model(user_song_list_count)
    
    # train surprise models if available
    surprise_results = None
    if not skip_surprise:
        from util import load_evaluation_data
        eval_df_merged = load_evaluation_data(unique_track_metadata_df)
        surprise_results = train_surprise_models(user_song_list_count, unique_track_metadata_df, eval_df_merged)
    
    return {
        'popularity_recs': popularity_recs,
        'is_model': is_model,
        'user_song_sub': user_song_sub,
        'svd_results': (U, S, Vt, small_set, urm),
        'surprise_results': surprise_results
    }
