"""
Advanced feature selection algorithms.

This module implements various feature selection techniques including
wrapper methods, filter methods, and hybrid approaches.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Callable
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    mutual_info_classif, f_classif, chi2, f_regression,
    SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import logging
from tqdm import tqdm
import warnings

logger = logging.getLogger(__name__)

class FeatureSelector:
    """
    Comprehensive feature selection toolkit.
    
    Implements multiple feature selection strategies:
    - Filter methods
    - Wrapper methods
    - Embedded methods
    - Hybrid methods
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.selected_features_ = None
        self.feature_scores_ = None
        self.selector_ = None
    
    def filter_methods(self, X: np.ndarray, y: np.ndarray, 
                      method: str = 'mutual_info', k: int = 10) -> np.ndarray:
        """
        Apply filter-based feature selection methods.
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Filter method ('mutual_info', 'f_classif', 'chi2', 'variance')
            k: Number of features to select
            
        Returns:
            Array of selected feature indices
        """
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'chi2':
            selector = SelectKBest(score_func=chi2, k=k)
        elif method == 'variance':
            selector = VarianceThreshold(threshold=0.01)
        else:
            raise ValueError(f"Unknown filter method: {method}")
        
        selector.fit(X, y)
        self.selector_ = selector
        self.selected_features_ = selector.get_support(indices=True)
        self.feature_scores_ = selector.scores_ if hasattr(selector, 'scores_') else None
        
        return self.selected_features_
    
    def wrapper_methods(self, X: np.ndarray, y: np.ndarray,
                       method: str = 'rfe', n_features: int = 10,
                       estimator=None) -> np.ndarray:
        """
        Apply wrapper-based feature selection methods.
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Wrapper method ('rfe', 'rfecv')
            n_features: Number of features to select
            estimator: Base estimator for wrapper methods
            
        Returns:
            Array of selected feature indices
        """
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        if method == 'rfe':
            selector = RFE(estimator=estimator, n_features_to_select=n_features)
        elif method == 'rfecv':
            selector = RFECV(estimator=estimator, cv=5, scoring='accuracy')
        else:
            raise ValueError(f"Unknown wrapper method: {method}")
        
        selector.fit(X, y)
        self.selector_ = selector
        self.selected_features_ = selector.get_support(indices=True)
        
        return self.selected_features_
    
    def embedded_methods(self, X: np.ndarray, y: np.ndarray,
                        method: str = 'lasso', alpha: float = 0.01) -> np.ndarray:
        """
        Apply embedded feature selection methods.
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Embedded method ('lasso', 'elastic_net', 'tree_based')
            alpha: Regularization parameter
            
        Returns:
            Array of selected feature indices
        """
        if method == 'lasso':
            estimator = LassoCV(cv=5, random_state=self.random_state)
        elif method == 'elastic_net':
            estimator = ElasticNetCV(cv=5, random_state=self.random_state)
        elif method == 'tree_based':
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown embedded method: {method}")
        
        selector = SelectFromModel(estimator=estimator)
        selector.fit(X, y)
        self.selector_ = selector
        self.selected_features_ = selector.get_support(indices=True)
        
        return self.selected_features_
    
    def hybrid_selection(self, X: np.ndarray, y: np.ndarray,
                        filter_method: str = 'mutual_info',
                        wrapper_method: str = 'rfe',
                        n_features_filter: int = 20,
                        n_features_final: int = 10) -> np.ndarray:
        """
        Apply hybrid feature selection combining filter and wrapper methods.
        
        Args:
            X: Feature matrix
            y: Target vector
            filter_method: Initial filter method
            wrapper_method: Final wrapper method
            n_features_filter: Number of features after filtering
            n_features_final: Final number of features
            
        Returns:
            Array of selected feature indices
        """
        # Step 1: Apply filter method
        filter_features = self.filter_methods(X, y, filter_method, n_features_filter)
        X_filtered = X[:, filter_features]
        
        # Step 2: Apply wrapper method on filtered features
        wrapper_selector = FeatureSelector(random_state=self.random_state)
        final_features = wrapper_selector.wrapper_methods(
            X_filtered, y, wrapper_method, n_features_final
        )
        
        # Map back to original feature indices
        self.selected_features_ = filter_features[final_features]
        
        return self.selected_features_

class SequentialFeatureSelector:
    """
    Sequential feature selection algorithms.
    """
    
    def __init__(self, estimator=None, scoring='accuracy', cv=5, random_state=42):
        self.estimator = estimator or RandomForestClassifier(random_state=random_state)
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.selected_features_ = []
        self.scores_ = []
    
    def forward_selection(self, X: np.ndarray, y: np.ndarray, 
                         n_features: int = 10) -> List[int]:
        """
        Forward sequential feature selection.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_features: Number of features to select
            
        Returns:
            List of selected feature indices
        """
        n_total_features = X.shape[1]
        remaining_features = list(range(n_total_features))
        selected_features = []
        
        for _ in tqdm(range(min(n_features, n_total_features)), desc="Forward Selection"):
            best_score = -np.inf
            best_feature = None
            
            for feature in remaining_features:
                # Create feature set with current feature
                current_features = selected_features + [feature]
                X_current = X[:, current_features]
                
                # Evaluate performance
                scores = cross_val_score(
                    self.estimator, X_current, y, 
                    cv=self.cv, scoring=self.scoring
                )
                score = np.mean(scores)
                
                if score > best_score:
                    best_score = score
                    best_feature = feature
            
            if best_feature is not None:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                self.scores_.append(best_score)
        
        self.selected_features_ = selected_features
        return selected_features
    
    def backward_elimination(self, X: np.ndarray, y: np.ndarray,
                           n_features: int = 10) -> List[int]:
        """
        Backward sequential feature elimination.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_features: Number of features to keep
            
        Returns:
            List of selected feature indices
        """
        n_total_features = X.shape[1]
        selected_features = list(range(n_total_features))
        
        while len(selected_features) > n_features:
            worst_score = np.inf
            worst_feature = None
            
            for feature in selected_features:
                # Create feature set without current feature
                current_features = [f for f in selected_features if f != feature]
                X_current = X[:, current_features]
                
                # Evaluate performance
                scores = cross_val_score(
                    self.estimator, X_current, y,
                    cv=self.cv, scoring=self.scoring
                )
                score = np.mean(scores)
                
                if score < worst_score:
                    worst_score = score
                    worst_feature = feature
            
            if worst_feature is not None:
                selected_features.remove(worst_feature)
                self.scores_.append(worst_score)
        
        self.selected_features_ = selected_features
        return selected_features

class FeatureSelectionEvaluator:
    """
    Comprehensive evaluation of feature selection methods.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results_ = {}
    
    def evaluate_methods(self, X: np.ndarray, y: np.ndarray,
                        methods: Dict[str, Dict] = None) -> Dict[str, Any]:
        """
        Evaluate multiple feature selection methods.
        
        Args:
            X: Feature matrix
            y: Target vector
            methods: Dictionary of methods to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        if methods is None:
            methods = {
                'mutual_info': {'type': 'filter', 'k': 10},
                'f_classif': {'type': 'filter', 'k': 10},
                'rfe': {'type': 'wrapper', 'n_features': 10},
                'lasso': {'type': 'embedded', 'alpha': 0.01},
                'tree_based': {'type': 'embedded'}
            }
        
        results = {}
        selector = FeatureSelector(random_state=self.random_state)
        
        for method_name, method_config in methods.items():
            try:
                logger.info(f"Evaluating method: {method_name}")
                
                if method_config['type'] == 'filter':
                    selected = selector.filter_methods(
                        X, y, method_name, method_config.get('k', 10)
                    )
                elif method_config['type'] == 'wrapper':
                    selected = selector.wrapper_methods(
                        X, y, method_name, method_config.get('n_features', 10)
                    )
                elif method_config['type'] == 'embedded':
                    selected = selector.embedded_methods(
                        X, y, method_name, method_config.get('alpha', 0.01)
                    )
                
                # Evaluate performance
                X_selected = X[:, selected]
                scores = cross_val_score(
                    RandomForestClassifier(random_state=self.random_state),
                    X_selected, y, cv=5, scoring='accuracy'
                )
                
                results[method_name] = {
                    'selected_features': selected,
                    'n_features': len(selected),
                    'accuracy_mean': np.mean(scores),
                    'accuracy_std': np.std(scores),
                    'feature_ratio': len(selected) / X.shape[1]
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {method_name}: {e}")
                results[method_name] = {
                    'error': str(e),
                    'selected_features': [],
                    'n_features': 0,
                    'accuracy_mean': 0.0,
                    'accuracy_std': 0.0,
                    'feature_ratio': 0.0
                }
        
        self.results_ = results
        return results
    
    def compare_methods(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Compare feature selection methods.
        
        Args:
            results: Results from evaluate_methods
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for method, result in results.items():
            if 'error' not in result:
                comparison_data.append({
                    'Method': method,
                    'N_Features': result['n_features'],
                    'Accuracy_Mean': result['accuracy_mean'],
                    'Accuracy_Std': result['accuracy_std'],
                    'Feature_Ratio': result['feature_ratio']
                })
        
        return pd.DataFrame(comparison_data).sort_values('Accuracy_Mean', ascending=False)
    
    def plot_comparison(self, results: Dict[str, Any], save_path: str = None):
        """Plot comparison of feature selection methods."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        df = self.compare_methods(results)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        sns.barplot(data=df, x='Method', y='Accuracy_Mean', ax=ax1)
        ax1.set_title('Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Feature count comparison
        sns.barplot(data=df, x='Method', y='N_Features', ax=ax2)
        ax2.set_title('Number of Selected Features')
        ax2.set_ylabel('N Features')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Utility functions
def get_feature_importance_scores(X: np.ndarray, y: np.ndarray, 
                                 method: str = 'mutual_info') -> np.ndarray:
    """Get feature importance scores using various methods."""
    if method == 'mutual_info':
        return mutual_info_classif(X, y)
    elif method == 'f_classif':
        return f_classif(X, y)[0]
    elif method == 'chi2':
        return chi2(X, y)[0]
    else:
        raise ValueError(f"Unknown method: {method}")

def create_feature_importance_plot(feature_names: List[str], 
                                 importance_scores: np.ndarray,
                                 top_k: int = 20, save_path: str = None):
    """Create a plot of feature importance scores."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get top k features
    top_indices = np.argsort(importance_scores)[-top_k:]
    top_names = [feature_names[i] for i in top_indices]
    top_scores = importance_scores[top_indices]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_scores, y=top_names)
    plt.title(f'Top {top_k} Feature Importance Scores')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=50, n_informative=20, random_state=42)
    
    # Initialize feature selector
    selector = FeatureSelector()
    
    # Apply different methods
    methods = ['mutual_info', 'f_classif', 'rfe', 'lasso']
    results = {}
    
    for method in methods:
        if method in ['mutual_info', 'f_classif']:
            selected = selector.filter_methods(X, y, method, k=10)
        elif method == 'rfe':
            selected = selector.wrapper_methods(X, y, method, n_features=10)
        elif method == 'lasso':
            selected = selector.embedded_methods(X, y, method)
        
        results[method] = selected
        print(f"{method}: Selected {len(selected)} features")
    
    # Evaluate methods
    evaluator = FeatureSelectionEvaluator()
    evaluation_results = evaluator.evaluate_methods(X, y)
    comparison_df = evaluator.compare_methods(evaluation_results)
    print("\nMethod Comparison:")
    print(comparison_df)