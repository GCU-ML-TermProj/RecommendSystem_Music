import os
import sys
import subprocess
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import svds
import math as mt

# suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

def check_and_fix_numpy():
    """check numpy version and downgrade if needed for surprise compatibility"""
    print("="*80)
    print("CHECKING NUMPY VERSION FOR SURPRISE COMPATIBILITY")
    print("="*80)
    
    numpy_version = np.__version__
    print(f"\nCurrent NumPy version: {numpy_version}")
    
    if numpy_version.startswith('2.'):
        print("\nNumPy 2.x detected!")
        print("scikit-surprise requires NumPy 1.x for compatibility.")
        print("\nDowngrading NumPy to version 1.x...")
        
        try:
            print("\nStep 1/3: Uninstalling NumPy 2.x...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', 'numpy', '-y'], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            print("Step 2/3: Installing NumPy 1.26.4...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy==1.26.4', 
                                 '--break-system-packages'], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            print("Step 3/3: Reinstalling scikit-surprise...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', 'scikit-surprise', '-y'], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-surprise', 
                                 '--break-system-packages', '--no-cache-dir'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            print("\n" + "="*80)
            print("NumPy DOWNGRADE SUCCESSFUL!")
            print("="*80)
            print("\nIMPORTANT: Please RESTART your runtime/kernel now!")
            print("\nAfter restarting, run this script again.")
            print("="*80)
            
            sys.exit(0)
            
        except Exception as e:
            print(f"\nAutomatic downgrade failed: {e}")
            print("\n" + "="*80)
            print("MANUAL INSTALLATION REQUIRED")
            print("="*80)
            print("\nPlease run these commands manually:")
            print("  pip uninstall numpy -y")
            print("  pip install numpy==1.26.4 --break-system-packages")
            print("  pip uninstall scikit-surprise -y")
            print("  pip install scikit-surprise --break-system-packages --no-cache-dir")
            print("\nThen restart your kernel and run this script again.")
            print("="*80)
            sys.exit(1)
    else:
        print(f"NumPy {numpy_version} is compatible with scikit-surprise")
    
    print("\n" + "="*80 + "\n")

def load_data():
    """load and preprocess music dataset"""
    print("\n--- Loading Dataset ---")
    
    # load raw data
    track_metadata_df = pd.read_csv('./song_data.csv')
    count_play_df = pd.read_csv('./10000.txt', sep='\t', header=None, 
                                names=['user','song','play_count'])
    
    print('First look at track metadata:')
    print('Number of rows:', track_metadata_df.shape[0])
    print('Number of unique songs:', len(track_metadata_df.song_id.unique()))
    print(track_metadata_df.head())
    print('\nPlay count data:')
    print(count_play_df.shape, '\n', count_play_df.head())
    
    # deduplicate track metadata
    unique_track_metadata_df = track_metadata_df.groupby('song_id').max().reset_index()
    print('\nNumber of rows after unique song id treatment:', unique_track_metadata_df.shape[0])
    print('Number of unique songs:', len(unique_track_metadata_df.song_id.unique()))
    print(unique_track_metadata_df.head())
    
    # merge datasets
    user_song_list_count = pd.merge(count_play_df, 
                                    unique_track_metadata_df, how='left', 
                                    left_on='song', 
                                    right_on='song_id')
    user_song_list_count.rename(columns={'play_count':'listen_count'}, inplace=True)
    del(user_song_list_count['song_id'])
    
    print('\nMerged dataset:')
    print(user_song_list_count.head())
    
    return user_song_list_count, count_play_df, unique_track_metadata_df

def load_evaluation_data(unique_track_metadata_df):
    """load kaggle evaluation data for testing"""
    try:
        eval_df = pd.read_csv('./kaggle_visible_evaluation_triplets.txt', sep='\t', 
                              header=None, names=['user','song','play_count'])
        print(f"Evaluation dataset loaded! Shape: {eval_df.shape}")
        
        # merge with metadata
        eval_df_merged = pd.merge(eval_df, unique_track_metadata_df, how='left', 
                                  left_on='song', right_on='song_id')
        eval_df_merged.rename(columns={'play_count':'listen_count'}, inplace=True)
        if 'song_id' in eval_df_merged.columns:
            del eval_df_merged['song_id']
        
        print(f"Evaluation data after merging with metadata: {eval_df_merged.shape}")
        return eval_df_merged
    except FileNotFoundError:
        print("kaggle_visible_evaluation_triplets.txt not found")
        return None

class popularity_recommender_py():
    """popularity-based recommender system"""
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_recommendations = None
        
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns={'user_id': 'score'}, inplace=True)
    
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending=[0,1])
    
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        
        self.popularity_recommendations = train_data_sort.head(10)

    def recommend(self, user_id):    
        user_recommendations = self.popularity_recommendations
        
        user_recommendations['user_id'] = user_id
    
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        
        return user_recommendations

class item_similarity_recommender_py():
    """item similarity based recommender system"""
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None
        
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        return user_items
        
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
        return item_users
        
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())
        return all_items
        
    def construct_cooccurence_matrix(self, user_songs, all_songs):
        """construct co-occurrence matrix for item similarity"""
        user_songs_users = []        
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))
            
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)
           
        for i in range(0, len(all_songs)):
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set(songs_i_data[self.user_id].unique())
            
            for j in range(0, len(user_songs)):       
                users_j = user_songs_users[j]
                users_intersection = users_i.intersection(users_j)
                
                if len(users_intersection) != 0:
                    users_union = users_i.union(users_j)
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0
        
        return cooccurence_matrix

    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        print("Non zero values in cooccurence_matrix: %d" % np.count_nonzero(cooccurence_matrix))
        
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
    
        columns = ['user_id', 'song', 'score', 'rank']
        df = pd.DataFrame(columns=columns)
         
        rank = 1 
        for i in range(0, len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)] = [user, all_songs[sort_index[i][1]], sort_index[i][0], rank]
                rank = rank + 1
        
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
 
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    def recommend(self, user):
        user_songs = self.get_user_items(user)    
        print("No. of unique songs for the user: %d" % len(user_songs))
        
        all_songs = self.get_all_items_train_data()
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
        
        return df_recommendations
    
    def get_similar_items(self, item_list):
        user_songs = item_list
        all_songs = self.get_all_items_train_data()
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
         
        return df_recommendations

def compute_svd(urm, K):
    """compute svd decomposition for matrix factorization"""
    U, s, Vt = svds(urm, K)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i,i] = mt.sqrt(s[i])

    U = csc_matrix(U, dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)
    
    return U, S, Vt

def compute_estimated_matrix(urm, U, S, Vt, uTest, K, test, MAX_UID, MAX_PID):
    """compute estimated ratings using svd"""
    rightTerm = S * Vt 
    max_recommendation = 250
    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    recomendRatings = np.zeros(shape=(MAX_UID, max_recommendation), dtype=np.float16)
    
    for userTest in uTest:
        prod = U[userTest, :] * rightTerm
        estimatedRatings[userTest, :] = prod.todense()
        recomendRatings[userTest, :] = (-estimatedRatings[userTest, :]).argsort()[:max_recommendation]
    
    return recomendRatings

def show_recomendations(uTest, uTest_recommended_items, small_set, num_recomendations=10):
    """display recommendations for test users"""
    for user in uTest:
        print('-'*70)
        print("Recommendation for user id {}".format(user))
        rank_value = 1
        i = 0
        while (rank_value < num_recomendations + 1):
            so = uTest_recommended_items[user, i:i+1][0]
            if (small_set.user[(small_set.so_index_value == so) & (small_set.us_index_value == user)].count() == 0):
                song_details = small_set[(small_set.so_index_value == so)].\
                    drop_duplicates('so_index_value')[['title','artist_name']]
                print("The number {} recommended song is {} BY {}".format(rank_value, 
                                                                      list(song_details['title'])[0],
                                                                      list(song_details['artist_name'])[0]))
                rank_value += 1
            i += 1

def create_popularity_recommendation(train_data, user_id, item_id, n=10):
    """create popularity based recommendations"""
    train_data_grouped = train_data.groupby([item_id]).agg({user_id: 'count'}).reset_index()
    train_data_grouped.rename(columns={user_id: 'score'}, inplace=True)
    
    train_data_sort = train_data_grouped.sort_values(['score', item_id], ascending=[0,1])
    train_data_sort['Rank'] = train_data_sort.score.rank(ascending=0, method='first')
    
    popularity_recommendations = train_data_sort.head(n)
    return popularity_recommendations

def check_surprise_library():
    """check and install surprise library if needed"""
    print("\n--- Importing Surprise Library ---")
    try:
        from surprise import KNNWithMeans, SVDpp, Dataset, Reader
        from surprise.model_selection import train_test_split as surprise_train_test_split
        from surprise import accuracy, Prediction
        print("Surprise library imported successfully!")
        return False
    except ImportError as e:
        print(f"Failed to import Surprise: {e}")
        print("\nTrying to install scikit-surprise...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-surprise', 
                                  '--break-system-packages', '--no-cache-dir'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            from surprise import KNNWithMeans, SVDpp, Dataset, Reader
            from surprise.model_selection import train_test_split as surprise_train_test_split
            from surprise import accuracy, Prediction
            print("Surprise installed and imported successfully!")
            return False
        except Exception as install_error:
            print(f"Could not install Surprise: {install_error}")
            print("\nSkipping Surprise-based models (KNN Item-Based CF and SVD++)")
            print("The script will continue with basic models only.")
            return True
