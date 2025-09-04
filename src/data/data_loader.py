"""
Intrusion Detection System (IDS) Data Loading Utilities.

This module provides functionality to load and prepare the KDD Cup 1999 dataset
for intrusion detection and feature selection experiments.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.datasets import load_wine, load_breast_cancer, load_iris, make_classification
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import logging
from urllib.request import urlretrieve
import zipfile

logger = logging.getLogger(__name__)

class IDSDataLoader:
    """
    Intrusion Detection System Data Loader.
    
    Specialized loader for KDD Cup 1999 dataset with attack type classification
    and network feature analysis capabilities.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Attack type mappings
        self.attack_types = {
            'normal': 0,
            'dos': 1,      # Denial of Service
            'probe': 2,    # Surveillance and probing
            'r2l': 3,      # Remote to Local
            'u2r': 4       # User to Root
        }
        
        # KDD Cup feature names (41 features)
        self.feature_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
        
        # Feature categories for analysis
        self.feature_categories = {
            'basic': [0, 1, 2, 3, 4, 5],  # Basic connection features
            'content': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  # Content features
            'traffic': [22, 23, 24, 25, 26, 27, 28, 29, 30, 31],  # Traffic features
            'host': [32, 33, 34, 35, 36, 37, 38, 39, 40]  # Host-based features
        }
    
    def load_kdd_dataset(self, n_samples: Optional[int] = None, 
                        attack_types: Optional[List[str]] = None,
                        binary_classification: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """
        Load KDD Cup 1999 dataset for intrusion detection.
        
        Args:
            n_samples: Number of samples to load (None for all)
            attack_types: List of attack types to include (None for all)
            binary_classification: If True, binary (normal vs attack), else multi-class
            
        Returns:
            Tuple of (X, y, feature_names, metadata)
        """
        logger.info("Loading KDD Cup 1999 dataset for intrusion detection...")
        
        # Download dataset if not exists
        train_file = os.path.join(self.data_dir, "KDDTrain+.txt")
        test_file = os.path.join(self.data_dir, "KDDTest+.txt")
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            self._download_kdd_cup()
        
        # Load training data
        df_train = pd.read_csv(train_file, header=None, nrows=n_samples)
        df_test = pd.read_csv(test_file, header=None, nrows=n_samples)
        
        # Combine datasets
        df = pd.concat([df_train, df_test], ignore_index=True)
        
        # Define columns
        columns = self.feature_names + ["label", "difficulty"]
        df.columns = columns
        
        # Remove difficulty column
        df = df.drop("difficulty", axis=1)
        
        # Filter attack types if specified
        if attack_types is not None:
            df = df[df['label'].isin(attack_types)]
        
        # Handle categorical features
        categorical_cols = ['protocol_type', 'service', 'flag']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        
        # Prepare features and target
        X = df[self.feature_names].values
        y_labels = df['label'].values
        
        # Convert labels to numeric
        if binary_classification:
            y = np.where(y_labels == 'normal', 0, 1)
            attack_mapping = {'normal': 0, 'attack': 1}
        else:
            # Multi-class classification
            unique_labels = np.unique(y_labels)
            attack_mapping = {label: i for i, label in enumerate(unique_labels)}
            y = np.array([attack_mapping[label] for label in y_labels])
        
        # Create metadata
        metadata = {
            'attack_mapping': attack_mapping,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
            'feature_categories': self.feature_categories
        }
        
        logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
        logger.info(f"Class distribution: {metadata['class_distribution']}")
        
        return X, y, self.feature_names, metadata
    
    def get_attack_statistics(self, y: np.ndarray, attack_mapping: Dict) -> Dict[str, Any]:
        """
        Get detailed attack statistics for the dataset.
        
        Args:
            y: Target labels
            attack_mapping: Mapping of labels to attack types
            
        Returns:
            Dictionary with attack statistics
        """
        unique, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        stats = {}
        for label, count in zip(unique, counts):
            attack_type = list(attack_mapping.keys())[list(attack_mapping.values()).index(label)]
            stats[attack_type] = {
                'count': int(count),
                'percentage': float(count / total_samples * 100),
                'label': int(label)
            }
        
        return stats
    
    def analyze_feature_importance_by_category(self, feature_importance: np.ndarray) -> Dict[str, float]:
        """
        Analyze feature importance by category.
        
        Args:
            feature_importance: Array of feature importance scores
            
        Returns:
            Dictionary with average importance by category
        """
        category_importance = {}
        for category, indices in self.feature_categories.items():
            category_importance[category] = float(np.mean(feature_importance[indices]))
        
        return category_importance
    
    def _download_kdd_cup(self):
        """Download KDD Cup 1999 dataset."""
        logger.info("Downloading KDD Cup 1999 dataset...")
        
        urls = {
            "KDDTrain+.txt": "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
            "KDDTest+.txt": "http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz"
        }
        
        for filename, url in urls.items():
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                try:
                    urlretrieve(url, filepath + ".gz")
                    with zipfile.ZipFile(filepath + ".gz", 'r') as zip_ref:
                        zip_ref.extractall(self.data_dir)
                    os.remove(filepath + ".gz")
                    logger.info(f"Downloaded {filename}")
                except Exception as e:
                    logger.error(f"Failed to download {filename}: {e}")
                    # Create dummy data if download fails
                    self._create_dummy_kdd_data(filepath)
    
    def _create_dummy_kdd_data(self, filepath: str):
        """Create dummy KDD data if download fails."""
        logger.info("Creating dummy KDD data...")
        
        n_samples = 10000
        n_features = 41
        
        # Generate random data
        X = np.random.randn(n_samples, n_features)
        
        # Add some categorical features
        X[:, 1] = np.random.randint(0, 3, n_samples)  # protocol_type
        X[:, 2] = np.random.randint(0, 70, n_samples)  # service
        X[:, 3] = np.random.randint(0, 2, n_samples)   # flag
        
        # Generate labels (80% normal, 20% attack)
        y = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        
        # Create DataFrame
        df = pd.DataFrame(X)
        df['label'] = np.where(y == 0, 'normal', 'attack')
        df['difficulty'] = np.random.randint(1, 4, n_samples)
        
        # Save to file
        df.to_csv(filepath, header=False, index=False)
        logger.info(f"Created dummy data: {filepath}")

class IDSDataPreprocessor:
    """
    Intrusion Detection System Data Preprocessor.
    
    Specialized preprocessor for network security data with attack-specific
    scaling and encoding strategies.
    """
    
    def __init__(self):
        self.scaler = None
        self.label_encoder = None
        self.is_fitted = False
        self.feature_ranges = None
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None, 
                     scaling: str = 'standard', robust_scaling: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform data for intrusion detection.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            scaling: Type of scaling ('standard', 'minmax', 'robust', 'none')
            robust_scaling: Use robust scaling for outlier-resistant preprocessing
            
        Returns:
            Tuple of (X_transformed, y_transformed)
        """
        # Store feature ranges for analysis
        self.feature_ranges = {
            'min': np.min(X, axis=0),
            'max': np.max(X, axis=0),
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0)
        }
        
        # Scale features
        if scaling == 'standard':
            self.scaler = StandardScaler()
        elif scaling == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling == 'robust' and robust_scaling:
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        else:
            self.scaler = None
        
        if self.scaler is not None:
            X_transformed = self.scaler.fit_transform(X)
        else:
            X_transformed = X.copy()
        
        # Encode labels if provided
        y_transformed = y
        if y is not None:
            if not np.issubdtype(y.dtype, np.number):
                self.label_encoder = LabelEncoder()
                y_transformed = self.label_encoder.fit_transform(y)
        
        self.is_fitted = True
        return X_transformed, y_transformed
    
    def transform(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Scale features
        if self.scaler is not None:
            X_transformed = self.scaler.transform(X)
        else:
            X_transformed = X.copy()
        
        # Encode labels if provided
        y_transformed = y
        if y is not None and self.label_encoder is not None:
            y_transformed = self.label_encoder.transform(y)
        
        return X_transformed, y_transformed
    
    def inverse_transform_labels(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform labels."""
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(y)
        return y

def load_and_preprocess_ids_dataset(n_samples: Optional[int] = None,
                                   test_size: float = 0.2,
                                   scaling: str = 'robust',
                                   binary_classification: bool = True,
                                   random_state: int = 42) -> Dict[str, Any]:
    """
    Load and preprocess KDD Cup 1999 dataset for intrusion detection.
    
    Args:
        n_samples: Number of samples to load (None for all)
        test_size: Fraction of data for testing
        scaling: Type of scaling to apply
        binary_classification: If True, binary (normal vs attack), else multi-class
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary containing train/test data and metadata
    """
    # Load dataset
    loader = IDSDataLoader()
    X, y, feature_names, metadata = loader.load_kdd_dataset(
        n_samples=n_samples,
        binary_classification=binary_classification
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Preprocess data
    preprocessor = IDSDataPreprocessor()
    X_train_scaled, y_train_encoded = preprocessor.fit_transform(X_train, y_train, scaling)
    X_test_scaled, y_test_encoded = preprocessor.transform(X_test, y_test)
    
    # Get attack statistics
    attack_stats = loader.get_attack_statistics(y, metadata['attack_mapping'])
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_encoded,
        'y_test': y_test_encoded,
        'feature_names': feature_names,
        'preprocessor': preprocessor,
        'metadata': metadata,
        'attack_stats': attack_stats,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'n_classes': len(np.unique(y))
    }

def create_feature_importance_plot(feature_names: List[str], 
                                 importance_scores: np.ndarray,
                                 top_k: int = 20) -> None:
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
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load KDD Cup dataset for intrusion detection
    data = load_and_preprocess_ids_dataset(n_samples=10000, binary_classification=True)
    
    print(f"Dataset shape: {data['X_train'].shape}")
    print(f"Number of classes: {data['n_classes']}")
    print(f"Feature names: {data['feature_names'][:5]}...")
    print(f"Attack statistics: {data['attack_stats']}")
    
    # Analyze feature importance by category
    loader = IDSDataLoader()
    dummy_importance = np.random.rand(41)  # Dummy importance scores
    category_importance = loader.analyze_feature_importance_by_category(dummy_importance)
    print(f"Feature importance by category: {category_importance}")