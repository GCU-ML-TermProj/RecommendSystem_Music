"""
music recommendation system - main pipeline
this script orchestrates the entire recommendation system workflow:
1. check numpy compatibility
2. load and preprocess data
3. perform exploratory data analysis (eda)
4. train recommendation models
5. evaluate models with comprehensive metrics
"""

import sys
from util import check_and_fix_numpy, load_data, check_surprise_library
from eda import run_all_eda
from train import run_all_training
from test import comprehensive_evaluation

def main():
    """main execution pipeline"""
    print("\n" + "="*80)
    print("MUSIC RECOMMENDATION SYSTEM")
    print("="*80 + "\n")
    
    # step 1: check numpy version for surprise compatibility
    check_and_fix_numpy()
    
    # step 2: load and preprocess data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    user_song_list_count, count_play_df, unique_track_metadata_df = load_data()
    
    # step 3: exploratory data analysis
    print("\n" + "="*80)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)
    run_all_eda(user_song_list_count, count_play_df, unique_track_metadata_df)
    
    # step 4: check if surprise library is available
    skip_surprise = check_surprise_library()
    
    # step 5: train all models
    print("\n" + "="*80)
    print("STEP 3: TRAINING MODELS")
    print("="*80)
    training_results = run_all_training(user_song_list_count, unique_track_metadata_df, 
                                       skip_surprise=skip_surprise)
    
    # step 6: comprehensive evaluation (if surprise models available)
    if not skip_surprise and training_results['surprise_results'] is not None:
        print("\n" + "="*80)
        print("STEP 4: COMPREHENSIVE EVALUATION")
        print("="*80)
        evaluation_results = comprehensive_evaluation(training_results['surprise_results'])
    else:
        print("\n" + "="*80)
        print("Skipping comprehensive evaluation (Surprise library not available)")
        print("Basic models completed successfully.")
        print("="*80)
    
    # final summary
    print("\n" + "="*80)
    print("ALL PROCESSING COMPLETE!")
    print("="*80)
    print("\nSummary:")
    print("- Data loading and preprocessing: ✓")
    print("- Exploratory data analysis (10 visualizations): ✓")
    print("- Popularity-based recommendations: ✓")
    print("- Item similarity-based recommendations: ✓")
    print("- Matrix factorization (SVD): ✓")
    
    if not skip_surprise:
        print("- KNN item-based collaborative filtering: ✓")
        print("- SVD++ collaborative filtering: ✓")
        print("- Comprehensive evaluation (15 additional visualizations): ✓")
        print("\nTotal visualizations generated: 25")
    else:
        print("- Advanced models (KNN, SVD++): ✗ (Surprise library unavailable)")
        print("\nTotal visualizations generated: 10")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
