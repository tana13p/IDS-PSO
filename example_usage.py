"""
Intrusion Detection System (IDS) Example Usage.

This script demonstrates how to use the IDS PSO framework
for network intrusion detection and feature selection.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
import time

# Import our IDS modules
from core.pso_optimizer import IDSPSOOptimizer, IDSPSOConfig, create_ids_pso_config
from data.data_loader import IDSDataLoader, load_and_preprocess_ids_dataset
from core.feature_selector import FeatureSelector, FeatureSelectionEvaluator

def main():
    """Main IDS example function."""
    print("🛡️ Intrusion Detection System (IDS) PSO Example")
    print("=" * 60)
    
    # 1. Load KDD Cup dataset for intrusion detection
    print("\n1. Loading KDD Cup 1999 dataset for intrusion detection...")
    data = load_and_preprocess_ids_dataset(
        n_samples=20000, 
        test_size=0.2, 
        scaling='robust',
        binary_classification=True
    )
    
    print(f"Dataset shape: {data['X_train'].shape}")
    print(f"Number of features: {data['n_features']}")
    print(f"Number of classes: {data['n_classes']}")
    print(f"Attack statistics: {data['attack_stats']}")
    print(f"Feature names: {data['feature_names'][:5]}...")
    
    # 2. Compare traditional feature selection methods for IDS
    print("\n2. Comparing traditional feature selection methods for IDS...")
    evaluator = FeatureSelectionEvaluator()
    
    methods = {
        'mutual_info': {'type': 'filter', 'k': 15},
        'f_classif': {'type': 'filter', 'k': 15},
        'rfe': {'type': 'wrapper', 'n_features': 15},
        'lasso': {'type': 'embedded', 'alpha': 0.01}
    }
    
    comparison_results = evaluator.evaluate_methods(
        data['X_train'], data['y_train'], methods
    )
    
    comparison_df = evaluator.compare_methods(comparison_results)
    print("\nTraditional Methods Comparison:")
    print(comparison_df)
    
    # 3. Run IDS PSO optimization
    print("\n3. Running IDS PSO optimization...")
    
    # Configure IDS PSO
    config = create_ids_pso_config(
        n_particles=30,
        n_iterations=50,
        parallel=True,
        classifier='rf',
        detection_rate_weight=0.7,
        false_positive_weight=0.3
    )
    
    # Initialize IDS optimizer
    ids_pso = IDSPSOOptimizer(config)
    
    # Run IDS optimization
    start_time = time.time()
    results = ids_pso.optimize_ids(
        data['X_train'], 
        data['y_train'], 
        data['feature_names'],
        data['attack_stats']
    )
    optimization_time = time.time() - start_time
    
    print(f"\nIDS PSO Optimization Results:")
    print(f"Selected features: {len(results['selected_features'])}")
    print(f"Detection Rate: {results['detection_rate']:.4f}")
    print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Optimization time: {optimization_time:.2f} seconds")
    print(f"Iterations: {results['n_iterations']}")
    print(f"Selected feature names: {results['selected_feature_names'][:5]}...")
    
    # 4. Evaluate IDS performance on test set
    print("\n4. Evaluating IDS performance on test set...")
    
    # Test on test set
    X_test_selected = data['X_test'][:, results['selected_features']]
    
    # Train final IDS model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Random Forest for IDS
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_test_selected, data['y_test'])
    rf_pred = rf.predict(X_test_selected)
    
    # Calculate IDS-specific metrics
    cm = confusion_matrix(data['y_test'], rf_pred)
    tn, fp, fn, tp = cm.ravel()
    
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = accuracy_score(data['y_test'], rf_pred)
    
    print(f"IDS Performance Metrics:")
    print(f"Detection Rate: {detection_rate:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:")
    print(f"  True Negatives: {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives: {tp}")
    
    # 5. Create IDS visualizations
    print("\n5. Creating IDS visualizations...")
    
    # Convergence plot
    plt.figure(figsize=(10, 6))
    plt.plot(results['fitness_history'])
    plt.title('IDS PSO Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (Security Score)')
    plt.grid(True)
    plt.savefig('results/ids_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance plot for IDS
    importance = results['feature_importance']
    top_indices = np.argsort(importance)[-15:]
    top_names = [data['feature_names'][i] for i in top_indices]
    top_scores = importance[top_indices]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_scores, y=top_names)
    plt.title('Top 15 Network Features for Intrusion Detection')
    plt.xlabel('Feature Importance Score')
    plt.tight_layout()
    plt.savefig('results/ids_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'], 
                yticklabels=['Normal', 'Attack'])
    plt.title('IDS Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/ids_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. IDS Performance comparison
    print("\n6. IDS Performance comparison...")
    
    # Compare with all features
    rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_all.fit(data['X_test'], data['y_test'])
    rf_all_pred = rf_all.predict(data['X_test'])
    rf_all_accuracy = accuracy_score(data['y_test'], rf_all_pred)
    
    # Calculate all-features IDS metrics
    cm_all = confusion_matrix(data['y_test'], rf_all_pred)
    tn_all, fp_all, fn_all, tp_all = cm_all.ravel()
    detection_rate_all = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0
    false_positive_rate_all = fp_all / (fp_all + tn_all) if (fp_all + tn_all) > 0 else 0
    
    print(f"\nIDS Performance Comparison:")
    print(f"All features ({data['n_features']}):")
    print(f"  Detection Rate: {detection_rate_all:.4f}")
    print(f"  False Positive Rate: {false_positive_rate_all:.4f}")
    print(f"  Accuracy: {rf_all_accuracy:.4f}")
    print(f"\nPSO selected ({len(results['selected_features'])}):")
    print(f"  Detection Rate: {detection_rate:.4f}")
    print(f"  False Positive Rate: {false_positive_rate:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"\nImprovements:")
    print(f"  Feature reduction: {(1 - len(results['selected_features'])/data['n_features'])*100:.1f}%")
    print(f"  Detection rate change: {(detection_rate - detection_rate_all)*100:+.2f}%")
    print(f"  False positive rate change: {(false_positive_rate - false_positive_rate_all)*100:+.2f}%")
    print(f"  Accuracy change: {(accuracy - rf_all_accuracy)*100:+.2f}%")
    
    # 7. IDS Method comparison summary
    print("\n7. IDS Method comparison summary...")
    
    # Add IDS PSO results to comparison
    ids_pso_result = {
        'Method': 'IDS PSO',
        'N_Features': len(results['selected_features']),
        'Detection_Rate': detection_rate,
        'False_Positive_Rate': false_positive_rate,
        'Accuracy': accuracy,
        'Feature_Ratio': len(results['selected_features']) / data['n_features']
    }
    
    # Create final IDS comparison
    final_comparison = comparison_df.copy()
    final_comparison = pd.concat([final_comparison, pd.DataFrame([ids_pso_result])], ignore_index=True)
    final_comparison = final_comparison.sort_values('Accuracy', ascending=False)
    
    print("\nFinal IDS Method Comparison:")
    print(final_comparison)
    
    # 8. Security Analysis
    print("\n8. Security Analysis...")
    
    # Analyze feature categories
    loader = IDSDataLoader()
    category_importance = loader.analyze_feature_importance_by_category(importance)
    
    print(f"\nFeature Importance by Category:")
    for category, score in category_importance.items():
        print(f"  {category.capitalize()}: {score:.4f}")
    
    # Security recommendations
    print(f"\nSecurity Recommendations:")
    print(f"  - Focus on {max(category_importance, key=category_importance.get)} features for detection")
    print(f"  - Monitor {len(results['selected_features'])} key network features")
    print(f"  - Achieved {detection_rate:.1%} attack detection rate")
    print(f"  - Reduced false positives to {false_positive_rate:.1%}")
    
    # 9. Save IDS results
    print("\n9. Saving IDS results...")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Save IDS optimization results
    results_df = pd.DataFrame({
        'feature_index': results['selected_features'],
        'feature_name': results['selected_feature_names'],
        'importance_score': results['feature_importance'][results['selected_features']]
    })
    
    results_df.to_csv('results/ids_selected_features.csv', index=False)
    
    # Save IDS comparison results
    final_comparison.to_csv('results/ids_method_comparison.csv', index=False)
    
    # Save IDS performance metrics
    performance_df = pd.DataFrame({
        'Metric': ['Detection Rate', 'False Positive Rate', 'Precision', 'Accuracy', 'F1 Score'],
        'Value': [detection_rate, false_positive_rate, precision, accuracy, results['f1_score']]
    })
    performance_df.to_csv('results/ids_performance_metrics.csv', index=False)
    
    print("IDS results saved to 'results/' directory")
    
    print("\n✅ IDS Example completed successfully!")
    print("\nNext steps:")
    print("- Run the IDS dashboard: python -m src.visualization.dashboard")
    print("- Start the IDS API server: python -m src.api.app")
    print("- Explore IDS analysis in the 'notebooks/' directory")
    print("- Deploy the IDS system using Docker: docker-compose up")

if __name__ == "__main__":
    main()