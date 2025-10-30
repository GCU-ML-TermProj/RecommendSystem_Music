import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Prediction

def calculate_metrics(predictions, k=10):
    """calculate comprehensive evaluation metrics"""
    user_est_true = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((est, true_r))
    
    precisions, recalls, f1_scores, ndcgs, hit_rates, mrrs = [], [], [], [], [], []
    
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]
        top_k_relevant = sum([1 for est, true_r in top_k if true_r >= 4])
        total_relevant = sum([1 for est, true_r in user_ratings if true_r >= 4])
        
        precision = top_k_relevant / k if k > 0 else 0
        precisions.append(precision)
        recall = top_k_relevant / total_relevant if total_relevant > 0 else 0
        recalls.append(recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        
        # calculate ndcg
        dcg = sum([(1 if true_r >= 4 else 0) / np.log2(i + 2) for i, (est, true_r) in enumerate(top_k)])
        idcg = sum([1 / np.log2(i + 2) for i in range(min(total_relevant, k))])
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
        
        hit_rates.append(1 if top_k_relevant > 0 else 0)
        
        # calculate mrr
        for i, (est, true_r) in enumerate(top_k):
            if true_r >= 4:
                mrrs.append(1 / (i + 1))
                break
        else:
            mrrs.append(0)
    
    return {
        'Precision@K': np.mean(precisions),
        'Recall@K': np.mean(recalls),
        'F1@K': np.mean(f1_scores),
        'NDCG@K': np.mean(ndcgs),
        'Hit_Rate@K': np.mean(hit_rates),
        'MRR': np.mean(mrrs)
    }

def calculate_coverage(predictions, all_items):
    """calculate item coverage"""
    recommended_items = set([iid for uid, iid, _, _, _ in predictions])
    return len(recommended_items) / len(all_items)

def calculate_diversity(predictions):
    """calculate recommendation diversity"""
    user_items = {}
    for uid, iid, _, _, _ in predictions:
        if uid not in user_items:
            user_items[uid] = set()
        user_items[uid].add(iid)
    return np.mean([len(items) for items in user_items.values()])

def plot_model_performance_comparison(knn_rmse, knn_mae, svdpp_rmse, svdpp_mae, 
                                     popularity_predictions, testset):
    """visualization 11: model performance comparison"""
    from surprise import accuracy
    
    fig = plt.figure(figsize=(12, 6))
    models = ['Popularity', 'KNN Item-Based', 'SVD++']
    
    # calculate popularity baseline
    pop_rmse = accuracy.rmse(popularity_predictions, verbose=False)
    pop_mae = accuracy.mae(popularity_predictions, verbose=False)
    
    rmse_scores = [pop_rmse, knn_rmse, svdpp_rmse]
    mae_scores = [pop_mae, knn_mae, svdpp_mae]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, rmse_scores, width, label='RMSE', color='steelblue')
    bars2 = plt.bar(x + width/2, mae_scores, width, label='MAE', color='coral')
    
    # add value labels
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Model')
    plt.ylabel('Error')
    plt.title('Model Performance Comparison (RMSE and MAE)')
    plt.xticks(x, models, rotation=15, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_prediction_error_distribution_knn(knn_predictions):
    """visualization 12: prediction error distribution (knn)"""
    fig = plt.figure(figsize=(12, 6))
    knn_errors = [abs(pred.r_ui - pred.est) for pred in knn_predictions]
    plt.hist(knn_errors, bins=50, edgecolor='black', color='lightblue', alpha=0.7)
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('KNN Item-Based CF: Prediction Error Distribution')
    plt.axvline(np.mean(knn_errors), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(knn_errors):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_prediction_error_distribution_svdpp(svdpp_predictions):
    """visualization 13: prediction error distribution (svd++)"""
    fig = plt.figure(figsize=(12, 6))
    svdpp_errors = [abs(pred.r_ui - pred.est) for pred in svdpp_predictions]
    plt.hist(svdpp_errors, bins=50, edgecolor='black', color='lightgreen', alpha=0.7)
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('SVD++: Prediction Error Distribution')
    plt.axvline(np.mean(svdpp_errors), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(svdpp_errors):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted_knn(knn_predictions):
    """visualization 14: actual vs predicted (knn)"""
    fig = plt.figure(figsize=(10, 10))
    actual_knn = [pred.r_ui for pred in knn_predictions[:1000]]
    predicted_knn = [pred.est for pred in knn_predictions[:1000]]
    plt.scatter(actual_knn, predicted_knn, alpha=0.5, color='blue')
    plt.plot([1, 5], [1, 5], 'r--', linewidth=2)
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title('KNN: Actual vs Predicted Ratings')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted_svdpp(svdpp_predictions):
    """visualization 15: actual vs predicted (svd++)"""
    fig = plt.figure(figsize=(10, 10))
    actual_svdpp = [pred.r_ui for pred in svdpp_predictions[:1000]]
    predicted_svdpp = [pred.est for pred in svdpp_predictions[:1000]]
    plt.scatter(actual_svdpp, predicted_svdpp, alpha=0.5, color='green')
    plt.plot([1, 5], [1, 5], 'r--', linewidth=2)
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title('SVD++: Actual vs Predicted Ratings')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_comprehensive_comparison(evaluation_results):
    """visualization 16: comprehensive comparison"""
    fig = plt.figure(figsize=(16, 8))
    metrics_to_plot = ['RMSE', 'MAE', 'Precision@10', 'Recall@10', 'F1@10', 'NDCG@10']
    x = np.arange(len(evaluation_results))
    width = 0.14
    
    for i, metric in enumerate(metrics_to_plot):
        bars = plt.bar(x + i*width - width*2.5, evaluation_results[metric], width, label=metric)
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Comprehensive Model Performance')
    plt.xticks(x, evaluation_results['Model'], rotation=15, ha='right')
    plt.legend(loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_precision_recall_tradeoff(evaluation_results):
    """visualization 17: precision-recall tradeoff"""
    fig = plt.figure(figsize=(10, 8))
    for model_name in evaluation_results['Model']:
        model_data = evaluation_results[evaluation_results['Model'] == model_name]
        plt.scatter(model_data['Recall@10'], model_data['Precision@10'], s=200, label=model_name)
        plt.annotate(f"{model_data['Recall@10'].values[0]:.4f}, {model_data['Precision@10'].values[0]:.4f}", 
                    (model_data['Recall@10'].values[0], model_data['Precision@10'].values[0]),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    plt.xlabel('Recall@10')
    plt.ylabel('Precision@10')
    plt.title('Precision-Recall Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_coverage_diversity(evaluation_results):
    """visualization 18: coverage & diversity"""
    fig = plt.figure(figsize=(12, 6))
    x = np.arange(len(evaluation_results))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, evaluation_results['Coverage'], width, label='Coverage', color='teal')
    bars2 = plt.bar(x + width/2, evaluation_results['Diversity']/100, width, label='Diversity (scaled)', color='orange')
    
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Coverage and Diversity Comparison')
    plt.xticks(x, evaluation_results['Model'], rotation=15, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_ranking_metrics(evaluation_results):
    """visualization 19: ranking metrics"""
    fig = plt.figure(figsize=(12, 6))
    ranking_metrics = ['NDCG@10', 'MRR', 'Hit_Rate@10']
    x = np.arange(len(evaluation_results))
    width = 0.25
    
    for i, metric in enumerate(ranking_metrics):
        bars = plt.bar(x + i*width - width, evaluation_results[metric], width, label=metric)
        for bar in bars:
            height = bar.get_height()
            if height > 0.001:
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Ranking Quality Metrics')
    plt.xticks(x, evaluation_results['Model'], rotation=15, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_error_analysis_heatmap(popularity_predictions, knn_predictions, svdpp_predictions):
    """visualization 20: error analysis heatmap"""
    fig = plt.figure(figsize=(18, 6))
    for idx, (predictions, model_name) in enumerate([
        (popularity_predictions, 'Popularity'),
        (knn_predictions, 'KNN'),
        (svdpp_predictions, 'SVD++')
    ]):
        plt.subplot(1, 3, idx+1)
        actual = [pred.r_ui for pred in predictions]
        predicted = [pred.est for pred in predictions]
        plt.hist2d(actual, predicted, bins=20, cmap='Blues')
        plt.colorbar(label='Frequency')
        plt.plot([1, 5], [1, 5], 'r--', linewidth=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{model_name}')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_per_user_rmse(svdpp_predictions):
    """visualization 21: per-user rmse distribution"""
    fig = plt.figure(figsize=(14, 6))
    user_rmses = {}
    for uid, iid, r_ui, est, _ in svdpp_predictions:
        if uid not in user_rmses:
            user_rmses[uid] = []
        user_rmses[uid].append((r_ui - est) ** 2)
    user_rmse_values = [np.sqrt(np.mean(errors)) for errors in user_rmses.values()]
    plt.hist(user_rmse_values, bins=50, edgecolor='black', color='mediumpurple', alpha=0.7)
    plt.xlabel('Per-User RMSE')
    plt.ylabel('Number of Users')
    plt.title('Distribution of Per-User RMSE (SVD++)')
    plt.axvline(np.mean(user_rmse_values), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(user_rmse_values):.3f}')
    plt.axvline(np.median(user_rmse_values), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(user_rmse_values):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_confidence_analysis(knn_predictions, svdpp_predictions):
    """visualization 22: confidence analysis"""
    fig = plt.figure(figsize=(16, 6))
    for idx, (predictions, model_name, color) in enumerate([
        (knn_predictions, 'KNN', 'steelblue'),
        (svdpp_predictions, 'SVD++', 'forestgreen')
    ]):
        plt.subplot(1, 2, idx+1)
        bins = [1, 2, 3, 4, 5]
        bin_errors = {i: [] for i in range(len(bins)-1)}
        for pred in predictions:
            for i in range(len(bins)-1):
                if bins[i] <= pred.est < bins[i+1]:
                    bin_errors[i].append(abs(pred.r_ui - pred.est))
                    break
        avg_errors = [np.mean(bin_errors[i]) if bin_errors[i] else 0 for i in range(len(bins)-1)]
        bin_labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)]
        bars = plt.bar(bin_labels, avg_errors, color=color, alpha=0.7, edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        plt.xlabel('Predicted Rating Range')
        plt.ylabel('Avg Absolute Error')
        plt.title(f'{model_name}: Error by Range')
        plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_activity_level_performance(svdpp_predictions, surprise_data):
    """visualization 23: activity level performance"""
    fig = plt.figure(figsize=(14, 6))
    user_activity = surprise_data.groupby('user').size()
    activity_quantiles = user_activity.quantile([0.33, 0.67])
    
    def get_activity_level(user):
        count = user_activity.get(user, 0)
        if count <= activity_quantiles.iloc[0]:
            return 'Low'
        elif count <= activity_quantiles.iloc[1]:
            return 'Medium'
        else:
            return 'High'
    
    activity_errors = {'Low': [], 'Medium': [], 'High': []}
    for uid, iid, r_ui, est, _ in svdpp_predictions:
        level = get_activity_level(uid)
        activity_errors[level].append((r_ui - est) ** 2)
    
    activity_rmses = {level: np.sqrt(np.mean(errors)) if errors else 0 
                      for level, errors in activity_errors.items()}
    colors_map = {'Low': 'lightcoral', 'Medium': 'lightskyblue', 'High': 'lightgreen'}
    bars = plt.bar(activity_rmses.keys(), activity_rmses.values(), 
            color=[colors_map[k] for k in activity_rmses.keys()], edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    plt.xlabel('User Activity Level')
    plt.ylabel('RMSE')
    plt.title('SVD++ Performance by Activity Level')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_top_k_quality(knn_predictions, svdpp_predictions):
    """visualization 24: top-k quality (2 charts for precision and recall)"""
    k_values = [5, 10, 15, 20, 25, 30]
    models_for_k = {
        'KNN': knn_predictions,
        'SVD++': svdpp_predictions
    }
    
    for metric_name, metric_key in [('Precision', 'precision'), ('Recall', 'recall')]:
        plt.figure(figsize=(10, 6))
        for model_name, predictions in models_for_k.items():
            metric_values = []
            for k in k_values:
                metrics = calculate_metrics(predictions, k=k)
                if metric_key == 'precision':
                    metric_values.append(metrics['Precision@K'])
                else:
                    metric_values.append(metrics['Recall@K'])
            plt.plot(k_values, metric_values, marker='o', linewidth=2, label=model_name)
        plt.xlabel('K')
        plt.ylabel(f'{metric_name}@K')
        plt.title(f'{metric_name}@K vs K')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_confusion_matrix(knn_predictions, svdpp_predictions):
    """visualization 25: confusion matrix"""
    fig = plt.figure(figsize=(18, 5))
    for idx, (predictions, model_name) in enumerate([
        (knn_predictions, 'KNN'),
        (svdpp_predictions, 'SVD++')
    ]):
        plt.subplot(1, 2, idx+1)
        confusion = np.zeros((5, 5))
        for pred in predictions:
            actual_bin = min(4, int(pred.r_ui) - 1)
            predicted_bin = min(4, max(0, int(pred.est) - 1))
            confusion[actual_bin][predicted_bin] += 1
        row_sums = confusion.sum(axis=1, keepdims=True)
        confusion_normalized = np.divide(confusion, row_sums, where=row_sums!=0, out=np.zeros_like(confusion))
        sns.heatmap(confusion_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5],
                   cbar_kws={'label': 'Proportion'})
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model_name}: Rating Matrix')
    plt.tight_layout()
    plt.show()

def comprehensive_evaluation(surprise_results):
    """run comprehensive evaluation with all visualizations"""
    from surprise import accuracy, Prediction
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION WITH IMPROVED TEST SET")
    print("="*80 + "\n")
    
    knn_predictions = surprise_results['knn_predictions']
    svdpp_predictions = surprise_results['svdpp_predictions']
    surprise_data = surprise_results['surprise_data']
    testset = surprise_results['testset']
    knn_rmse = surprise_results['knn_rmse']
    knn_mae = surprise_results['knn_mae']
    svdpp_rmse = surprise_results['svdpp_rmse']
    svdpp_mae = surprise_results['svdpp_mae']
    
    # calculate popularity baseline
    popular_songs_ratings = surprise_data.groupby('song')['rating'].mean().to_dict()
    overall_mean = surprise_data['rating'].mean()
    popularity_predictions = []
    for uid, iid, r_ui in testset:
        est = popular_songs_ratings.get(iid, overall_mean)
        popularity_predictions.append(Prediction(uid, iid, r_ui, est, {}))
    pop_rmse = accuracy.rmse(popularity_predictions, verbose=False)
    pop_mae = accuracy.mae(popularity_predictions, verbose=False)
    
    # calculate all metrics
    pop_metrics = calculate_metrics(popularity_predictions)
    knn_metrics = calculate_metrics(knn_predictions)
    svdpp_metrics = calculate_metrics(svdpp_predictions)
    
    all_items = surprise_data.song.unique()
    
    # create evaluation results dataframe
    evaluation_results = pd.DataFrame({
        'Model': ['Popularity-Based', 'KNN Item-Based', 'SVD++'],
        'RMSE': [pop_rmse, knn_rmse, svdpp_rmse],
        'MAE': [pop_mae, knn_mae, svdpp_mae],
        'Precision@10': [pop_metrics['Precision@K'], knn_metrics['Precision@K'], svdpp_metrics['Precision@K']],
        'Recall@10': [pop_metrics['Recall@K'], knn_metrics['Recall@K'], svdpp_metrics['Recall@K']],
        'F1@10': [pop_metrics['F1@K'], knn_metrics['F1@K'], svdpp_metrics['F1@K']],
        'NDCG@10': [pop_metrics['NDCG@K'], knn_metrics['NDCG@K'], svdpp_metrics['NDCG@K']],
        'Hit_Rate@10': [pop_metrics['Hit_Rate@K'], knn_metrics['Hit_Rate@K'], svdpp_metrics['Hit_Rate@K']],
        'MRR': [pop_metrics['MRR'], knn_metrics['MRR'], svdpp_metrics['MRR']],
        'Coverage': [
            calculate_coverage(popularity_predictions, all_items),
            calculate_coverage(knn_predictions, all_items),
            calculate_coverage(svdpp_predictions, all_items)
        ],
        'Diversity': [
            calculate_diversity(popularity_predictions),
            calculate_diversity(knn_predictions),
            calculate_diversity(svdpp_predictions)
        ]
    })
    
    print("\n--- COMPREHENSIVE EVALUATION RESULTS ---")
    print(evaluation_results)
    
    # generate all visualizations (11-25)
    print("\n--- Generating Evaluation Visualizations (11-25) ---\n")
    
    plot_model_performance_comparison(knn_rmse, knn_mae, svdpp_rmse, svdpp_mae, 
                                     popularity_predictions, testset)
    plot_prediction_error_distribution_knn(knn_predictions)
    plot_prediction_error_distribution_svdpp(svdpp_predictions)
    plot_actual_vs_predicted_knn(knn_predictions)
    plot_actual_vs_predicted_svdpp(svdpp_predictions)
    plot_comprehensive_comparison(evaluation_results)
    plot_precision_recall_tradeoff(evaluation_results)
    plot_coverage_diversity(evaluation_results)
    plot_ranking_metrics(evaluation_results)
    plot_error_analysis_heatmap(popularity_predictions, knn_predictions, svdpp_predictions)
    plot_per_user_rmse(svdpp_predictions)
    plot_confidence_analysis(knn_predictions, svdpp_predictions)
    plot_activity_level_performance(svdpp_predictions, surprise_data)
    plot_top_k_quality(knn_predictions, svdpp_predictions)
    plot_confusion_matrix(knn_predictions, svdpp_predictions)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nTotal Visualizations: 25")
    print(f"\nFinal Ranking (by RMSE):")
    ranked = evaluation_results.sort_values('RMSE')
    for i, row in ranked.iterrows():
        print(f"{row['Model']}: RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}, NDCG@10={row['NDCG@10']:.4f}")
    
    return evaluation_results
