"""
Intrusion Detection System (IDS) PSO Optimizer

This module implements an enhanced PSO algorithm specifically designed for
intrusion detection feature selection, with security-focused metrics and
real-time optimization capabilities.
"""

import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import time
import warnings
from tqdm import tqdm
import joblib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IDSPSOConfig:
    """Configuration class for IDS PSO parameters."""
    n_particles: int = 50
    n_iterations: int = 100
    w: float = 0.9  # Inertia weight
    c1: float = 2.0  # Cognitive parameter
    c2: float = 2.0  # Social parameter
    v_max: float = 4.0  # Maximum velocity
    parallel: bool = True
    n_jobs: int = -1
    early_stopping: bool = True
    patience: int = 10
    min_features: int = 5  # Minimum features for IDS
    max_features: Optional[int] = 25  # Maximum features for IDS
    classifier: str = 'rf'  # Random Forest works well for IDS
    cv_folds: int = 5
    random_state: int = 42
    # IDS-specific parameters
    false_positive_weight: float = 0.3  # Weight for false positive penalty
    detection_rate_weight: float = 0.7  # Weight for detection rate
    feature_cost_weight: float = 0.1  # Weight for feature count penalty
    attack_type_weights: Dict[str, float] = None  # Weights for different attack types

class IDSParticle:
    """Individual particle in the swarm for IDS optimization."""
    
    def __init__(self, n_features: int, config: IDSPSOConfig):
        self.position = np.random.random(n_features)
        self.velocity = np.random.uniform(-1, 1, n_features)
        self.best_position = self.position.copy()
        self.best_fitness = -np.inf
        self.fitness = -np.inf
        
    def update_velocity(self, global_best: np.ndarray, config: PSOConfig):
        """Update particle velocity using PSO equations."""
        r1, r2 = np.random.random(2)
        
        cognitive = config.c1 * r1 * (self.best_position - self.position)
        social = config.c2 * r2 * (global_best - self.position)
        
        self.velocity = (config.w * self.velocity + cognitive + social)
        self.velocity = np.clip(self.velocity, -config.v_max, config.v_max)
    
    def update_position(self, config: PSOConfig):
        """Update particle position."""
        self.position += self.velocity
        self.position = np.clip(self.position, 0, 1)
    
    def get_selected_features(self, threshold: float = 0.5) -> List[int]:
        """Get indices of selected features based on threshold."""
        return [i for i, val in enumerate(self.position) if val >= threshold]

class IDSPSOOptimizer:
    """
    Intrusion Detection System PSO Optimizer.
    
    This class implements PSO specifically designed for intrusion detection
    feature selection with security-focused metrics and real-time optimization.
    """
    
    def __init__(self, config: IDSPSOConfig = None):
        self.config = config or IDSPSOConfig()
        self.particles: List[IDSParticle] = []
        self.global_best_position: np.ndarray = None
        self.global_best_fitness: float = -np.inf
        self.fitness_history: List[float] = []
        self.feature_importance: np.ndarray = None
        self.attack_detection_rates: Dict[str, float] = {}
        self.false_positive_rates: Dict[str, float] = {}
        self.classifier_map = {
            'mlp': MLPClassifier,
            'rf': RandomForestClassifier,
            'svm': SVC,
            'lr': LogisticRegression
        }
        
    def _initialize_particles(self, n_features: int):
        """Initialize particle swarm for IDS optimization."""
        self.particles = [
            IDSParticle(n_features, self.config) 
            for _ in range(self.config.n_particles)
        ]
        
    def _get_classifier(self):
        """Get configured classifier."""
        classifier_class = self.classifier_map[self.config.classifier]
        
        if self.config.classifier == 'mlp':
            return classifier_class(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.config.random_state
            )
        elif self.config.classifier == 'rf':
            return classifier_class(
                n_estimators=100,
                random_state=self.config.random_state
            )
        elif self.config.classifier == 'svm':
            return classifier_class(
                kernel='rbf',
                random_state=self.config.random_state
            )
        else:  # lr
            return classifier_class(
                random_state=self.config.random_state
            )
    
    def _evaluate_ids_fitness(self, particle: IDSParticle, X: np.ndarray, y: np.ndarray, 
                             attack_stats: Dict = None) -> float:
        """Evaluate fitness of a particle for IDS optimization."""
        selected_features = particle.get_selected_features()
        
        if len(selected_features) < self.config.min_features:
            return 0.0
            
        if self.config.max_features and len(selected_features) > self.config.max_features:
            return 0.0
        
        if len(selected_features) == 0:
            return 0.0
            
        try:
            X_selected = X[:, selected_features]
            classifier = self._get_classifier()
            
            # Use cross-validation for robust evaluation
            scores = cross_val_score(
                classifier, X_selected, y,
                cv=self.config.cv_folds,
                scoring='accuracy',
                n_jobs=1
            )
            
            # Calculate IDS-specific metrics
            y_pred = classifier.fit(X_selected, y).predict(X_selected)
            cm = confusion_matrix(y, y_pred)
            
            # Calculate detection rate (recall for attack class)
            if cm.shape[0] == 2:  # Binary classification
                tn, fp, fn, tp = cm.ravel()
                detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
                false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            else:
                # Multi-class: use macro average
                detection_rate = recall_score(y, y_pred, average='macro')
                false_positive_rate = 1 - precision_score(y, y_pred, average='macro')
            
            # IDS-specific fitness function
            accuracy = np.mean(scores)
            
            # Weighted fitness considering IDS priorities
            fitness = (
                self.config.detection_rate_weight * detection_rate +
                (1 - self.config.false_positive_weight) * (1 - false_positive_rate) +
                accuracy * 0.3
            )
            
            # Feature cost penalty
            feature_penalty = self.config.feature_cost_weight * (len(selected_features) / X.shape[1])
            fitness -= feature_penalty
            
            return max(0.0, fitness)
            
        except Exception as e:
            logger.warning(f"IDS fitness evaluation failed: {e}")
            return 0.0
    
    def _evaluate_particles_parallel(self, X: np.ndarray, y: np.ndarray, 
                                   attack_stats: Dict = None) -> List[float]:
        """Evaluate all particles in parallel for IDS optimization."""
        if not self.config.parallel:
            return [self._evaluate_ids_fitness(p, X, y, attack_stats) for p in self.particles]
        
        n_jobs = self.config.n_jobs if self.config.n_jobs != -1 else mp.cpu_count()
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(self._evaluate_ids_fitness, p, X, y, attack_stats) 
                for p in self.particles
            ]
            fitness_scores = [future.result() for future in futures]
        
        return fitness_scores
    
    def _update_global_best(self):
        """Update global best particle."""
        for particle in self.particles:
            if particle.fitness > self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
    
    def _calculate_security_metrics(self, X: np.ndarray, y: np.ndarray, 
                                  selected_features: List[int]) -> Dict[str, float]:
        """Calculate security-specific metrics for IDS evaluation."""
        if len(selected_features) == 0:
            return {}
        
        try:
            X_selected = X[:, selected_features]
            classifier = self._get_classifier()
            classifier.fit(X_selected, y)
            y_pred = classifier.predict(X_selected)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y, y_pred)
            
            if cm.shape[0] == 2:  # Binary classification
                tn, fp, fn, tp = cm.ravel()
                detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
                false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1 = 2 * (precision * detection_rate) / (precision + detection_rate) if (precision + detection_rate) > 0 else 0
            else:
                # Multi-class metrics
                detection_rate = recall_score(y, y_pred, average='macro')
                false_positive_rate = 1 - precision_score(y, y_pred, average='macro')
                precision = precision_score(y, y_pred, average='macro')
                f1 = f1_score(y, y_pred, average='macro')
            
            return {
                'detection_rate': detection_rate,
                'false_positive_rate': false_positive_rate,
                'precision': precision,
                'f1_score': f1,
                'accuracy': accuracy_score(y, y_pred)
            }
        except Exception as e:
            logger.warning(f"Security metrics calculation failed: {e}")
            return {}
    
    def _adaptive_parameters(self, iteration: int) -> Tuple[float, float, float]:
        """Adapt PSO parameters based on iteration."""
        # Linear decrease in inertia weight
        w = self.config.w * (1 - iteration / self.config.n_iterations)
        
        # Adaptive cognitive and social parameters
        c1 = self.config.c1 * (1 - iteration / (2 * self.config.n_iterations))
        c2 = self.config.c2 * (1 + iteration / (2 * self.config.n_iterations))
        
        return w, c1, c2
    
    def optimize_ids(self, X: np.ndarray, y: np.ndarray, 
                     feature_names: Optional[List[str]] = None,
                     attack_stats: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run PSO optimization for IDS feature selection.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Optional list of feature names
            attack_stats: Attack statistics for weighted optimization
            
        Returns:
            Dictionary containing IDS optimization results
        """
        logger.info("Starting IDS PSO optimization...")
        start_time = time.time()
        
        n_features = X.shape[1]
        self._initialize_particles(n_features)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Initialize global best
        self.global_best_position = self.particles[0].position.copy()
        
        # Main optimization loop
        no_improvement_count = 0
        
        for iteration in tqdm(range(self.config.n_iterations), desc="IDS PSO Optimization"):
            # Adaptive parameters
            w, c1, c2 = self._adaptive_parameters(iteration)
            
            # Evaluate particles with IDS-specific fitness
            fitness_scores = self._evaluate_particles_parallel(X, y, attack_stats)
            
            # Update particles
            for i, particle in enumerate(self.particles):
                particle.fitness = fitness_scores[i]
                
                # Update personal best
                if particle.fitness > particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                
                # Update velocity and position
                particle.update_velocity(self.global_best_position, 
                                       IDSPSOConfig(w=w, c1=c1, c2=c2))
                particle.update_position(self.config)
            
            # Update global best
            self._update_global_best()
            self.fitness_history.append(self.global_best_fitness)
            
            # Early stopping
            if self.config.early_stopping:
                if iteration > 0 and self.fitness_history[-1] <= self.fitness_history[-2]:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                
                if no_improvement_count >= self.config.patience:
                    logger.info(f"Early stopping at iteration {iteration}")
                    break
            
            # Log progress
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Best fitness = {self.global_best_fitness:.4f}")
        
        # Final evaluation
        selected_features = self.global_best_position >= 0.5
        selected_indices = np.where(selected_features)[0]
        
        # Calculate feature importance
        self.feature_importance = self.global_best_position
        
        # Calculate security metrics
        security_metrics = self._calculate_security_metrics(X, y, selected_indices.tolist())
        
        # Prepare IDS-specific results
        results = {
            'selected_features': selected_indices.tolist(),
            'selected_feature_names': [feature_names[i] for i in selected_indices],
            'best_fitness': self.global_best_fitness,
            'n_selected_features': len(selected_indices),
            'feature_importance': self.feature_importance,
            'fitness_history': self.fitness_history,
            'optimization_time': time.time() - start_time,
            'n_iterations': iteration + 1,
            'security_metrics': security_metrics,
            'detection_rate': security_metrics.get('detection_rate', 0.0),
            'false_positive_rate': security_metrics.get('false_positive_rate', 0.0),
            'precision': security_metrics.get('precision', 0.0),
            'f1_score': security_metrics.get('f1_score', 0.0),
            'accuracy': security_metrics.get('accuracy', 0.0)
        }
        
        logger.info(f"IDS optimization completed in {results['optimization_time']:.2f} seconds")
        logger.info(f"Selected {len(selected_indices)} features out of {n_features}")
        logger.info(f"Detection Rate: {results['detection_rate']:.4f}")
        logger.info(f"False Positive Rate: {results['false_positive_rate']:.4f}")
        
        return results
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.feature_importance
    
    def plot_convergence(self, save_path: Optional[str] = None):
        """Plot convergence curve."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history)
        plt.title('PSO Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class MultiObjectivePSO(PSOOptimizer):
    """
    Multi-objective PSO for feature selection.
    
    Optimizes both classification accuracy and number of features.
    """
    
    def __init__(self, config: PSOConfig = None):
        super().__init__(config)
        self.pareto_front: List[Particle] = []
    
    def _evaluate_fitness(self, particle: Particle, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate multi-objective fitness."""
        selected_features = particle.get_selected_features()
        
        if len(selected_features) < self.config.min_features:
            return (0.0, 1.0)  # (accuracy, feature_ratio)
        
        try:
            X_selected = X[:, selected_features]
            classifier = self._get_classifier()
            
            scores = cross_val_score(
                classifier, X_selected, y,
                cv=self.config.cv_folds,
                scoring='accuracy',
                n_jobs=1
            )
            
            accuracy = np.mean(scores)
            feature_ratio = len(selected_features) / X.shape[1]
            
            return (accuracy, feature_ratio)
            
        except Exception as e:
            logger.warning(f"Multi-objective fitness evaluation failed: {e}")
            return (0.0, 1.0)
    
    def _dominates(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """Check if p1 dominates p2."""
        return (p1[0] >= p2[0] and p1[1] <= p2[1]) and (p1[0] > p2[0] or p1[1] < p2[1])
    
    def _update_pareto_front(self):
        """Update Pareto front."""
        self.pareto_front = []
        
        for particle in self.particles:
            is_dominated = False
            for other in self.particles:
                if self._dominates(other.fitness, particle.fitness):
                    is_dominated = True
                    break
            
            if not is_dominated:
                self.pareto_front.append(particle)

# Utility functions
def create_ids_pso_config(n_particles: int = 50, n_iterations: int = 100, 
                         parallel: bool = True, classifier: str = 'rf',
                         detection_rate_weight: float = 0.7,
                         false_positive_weight: float = 0.3) -> IDSPSOConfig:
    """Create an IDS PSO configuration with security-focused parameters."""
    return IDSPSOConfig(
        n_particles=n_particles,
        n_iterations=n_iterations,
        parallel=parallel,
        classifier=classifier,
        detection_rate_weight=detection_rate_weight,
        false_positive_weight=false_positive_weight,
        min_features=5,
        max_features=25
    )

def compare_classifiers(X: np.ndarray, y: np.ndarray, 
                       selected_features: List[int]) -> Dict[str, float]:
    """Compare different classifiers on selected features."""
    X_selected = X[:, selected_features]
    results = {}
    
    classifiers = {
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    for name, clf in classifiers.items():
        try:
            scores = cross_val_score(clf, X_selected, y, cv=5, scoring='accuracy')
            results[name] = np.mean(scores)
        except Exception as e:
            logger.warning(f"Failed to evaluate {name}: {e}")
            results[name] = 0.0
    
    return results

# Example usage for IDS
if __name__ == "__main__":
    from src.data.data_loader import load_and_preprocess_ids_dataset
    
    # Load KDD Cup dataset for intrusion detection
    data = load_and_preprocess_ids_dataset(n_samples=10000, binary_classification=True)
    
    # Create IDS PSO configuration
    config = create_ids_pso_config(
        n_particles=30, 
        n_iterations=50, 
        parallel=True,
        detection_rate_weight=0.7,
        false_positive_weight=0.3
    )
    
    # Initialize IDS optimizer
    ids_pso = IDSPSOOptimizer(config)
    
    # Run IDS optimization
    results = ids_pso.optimize_ids(
        data['X_train'], 
        data['y_train'], 
        data['feature_names'],
        data['attack_stats']
    )
    
    print(f"Selected features: {results['selected_features']}")
    print(f"Detection Rate: {results['detection_rate']:.4f}")
    print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Optimization time: {results['optimization_time']:.2f}s")
    
    # Plot convergence
    ids_pso.plot_convergence()