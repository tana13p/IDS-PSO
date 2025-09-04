"""
REST API for PSO feature selection.

This module provides a RESTful API for running PSO optimization
and feature selection tasks.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import json
import traceback
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pso_optimizer import PSOOptimizer, PSOConfig, create_pso_config
from data.data_loader import DataLoader, load_and_preprocess_dataset
from core.feature_selector import FeatureSelector, FeatureSelectionEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for caching
cached_data = {}
optimization_cache = {}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """List available datasets."""
    try:
        loader = DataLoader()
        datasets = loader.list_available_datasets()
        return jsonify({
            'datasets': datasets,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/datasets/<dataset_name>', methods=['POST'])
def load_dataset(dataset_name: str):
    """Load a specific dataset."""
    try:
        # Get parameters from request
        params = request.get_json() or {}
        
        # Load dataset
        data = load_and_preprocess_dataset(
            dataset_name,
            test_size=params.get('test_size', 0.2),
            scaling=params.get('scaling', 'standard'),
            random_state=params.get('random_state', 42)
        )
        
        # Cache the data
        cache_key = f"{dataset_name}_{hash(str(params))}"
        cached_data[cache_key] = data
        
        return jsonify({
            'cache_key': cache_key,
            'n_features': data['n_features'],
            'n_samples': data['n_samples'],
            'n_classes': data['n_classes'],
            'feature_names': data['feature_names'][:10],  # First 10 features
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/optimize', methods=['POST'])
def run_optimization():
    """Run PSO optimization."""
    try:
        # Get parameters from request
        params = request.get_json()
        
        if not params:
            return jsonify({
                'error': 'No parameters provided',
                'status': 'error'
            }), 400
        
        # Extract parameters
        cache_key = params.get('cache_key')
        if not cache_key or cache_key not in cached_data:
            return jsonify({
                'error': 'Dataset not found. Please load a dataset first.',
                'status': 'error'
            }), 400
        
        data = cached_data[cache_key]
        
        # PSO configuration
        pso_config = PSOConfig(
            n_particles=params.get('n_particles', 50),
            n_iterations=params.get('n_iterations', 100),
            classifier=params.get('classifier', 'mlp'),
            parallel=params.get('parallel', True),
            n_jobs=params.get('n_jobs', -1),
            random_state=params.get('random_state', 42)
        )
        
        # Initialize optimizer
        optimizer = PSOOptimizer(pso_config)
        
        # Run optimization
        X_train = data['X_train']
        y_train = data['y_train']
        feature_names = data['feature_names']
        
        results = optimizer.optimize(X_train, y_train, feature_names)
        
        # Cache results
        optimization_cache[cache_key] = {
            'results': results,
            'optimizer': optimizer,
            'timestamp': datetime.now().isoformat()
        }
        
        # Prepare response
        response = {
            'selected_features': results['selected_features'],
            'selected_feature_names': results['selected_feature_names'],
            'best_fitness': results['best_fitness'],
            'n_selected_features': results['n_selected_features'],
            'optimization_time': results['optimization_time'],
            'n_iterations': results['n_iterations'],
            'fitness_history': results['fitness_history'],
            'feature_importance': results['feature_importance'].tolist(),
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error running optimization: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate_features():
    """Evaluate selected features."""
    try:
        params = request.get_json()
        
        if not params:
            return jsonify({
                'error': 'No parameters provided',
                'status': 'error'
            }), 400
        
        cache_key = params.get('cache_key')
        if not cache_key or cache_key not in cached_data:
            return jsonify({
                'error': 'Dataset not found. Please load a dataset first.',
                'status': 'error'
            }), 400
        
        data = cached_data[cache_key]
        selected_features = params.get('selected_features', [])
        
        if not selected_features:
            return jsonify({
                'error': 'No features selected',
                'status': 'error'
            }), 400
        
        # Evaluate on test set
        X_test = data['X_test']
        y_test = data['y_test']
        X_test_selected = X_test[:, selected_features]
        
        # Use different classifiers
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        classifiers = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'MLP': MLPClassifier(random_state=42),
            'SVM': SVC(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42)
        }
        
        results = {}
        for name, clf in classifiers.items():
            try:
                clf.fit(X_test_selected, y_test)
                y_pred = clf.predict(X_test_selected)
                
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1_score': f1_score(y_test, y_pred, average='weighted')
                }
            except Exception as e:
                logger.warning(f"Error evaluating {name}: {e}")
                results[name] = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0
                }
        
        return jsonify({
            'evaluation_results': results,
            'n_features': len(selected_features),
            'feature_ratio': len(selected_features) / data['n_features'],
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error evaluating features: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/compare', methods=['POST'])
def compare_methods():
    """Compare different feature selection methods."""
    try:
        params = request.get_json()
        
        if not params:
            return jsonify({
                'error': 'No parameters provided',
                'status': 'error'
            }), 400
        
        cache_key = params.get('cache_key')
        if not cache_key or cache_key not in cached_data:
            return jsonify({
                'error': 'Dataset not found. Please load a dataset first.',
                'status': 'error'
            }), 400
        
        data = cached_data[cache_key]
        X_train = data['X_train']
        y_train = data['y_train']
        
        # Initialize evaluator
        evaluator = FeatureSelectionEvaluator()
        
        # Define methods to compare
        methods = {
            'mutual_info': {'type': 'filter', 'k': 10},
            'f_classif': {'type': 'filter', 'k': 10},
            'rfe': {'type': 'wrapper', 'n_features': 10},
            'lasso': {'type': 'embedded', 'alpha': 0.01},
            'tree_based': {'type': 'embedded'}
        }
        
        # Run comparison
        results = evaluator.evaluate_methods(X_train, y_train, methods)
        comparison_df = evaluator.compare_methods(results)
        
        # Convert to JSON-serializable format
        comparison_data = comparison_df.to_dict('records')
        
        return jsonify({
            'comparison_results': comparison_data,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error comparing methods: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/visualize', methods=['POST'])
def create_visualizations():
    """Create visualization data for the optimization results."""
    try:
        params = request.get_json()
        
        if not params:
            return jsonify({
                'error': 'No parameters provided',
                'status': 'error'
            }), 400
        
        cache_key = params.get('cache_key')
        if not cache_key or cache_key not in optimization_cache:
            return jsonify({
                'error': 'Optimization results not found. Please run optimization first.',
                'status': 'error'
            }), 400
        
        optimization_data = optimization_cache[cache_key]
        results = optimization_data['results']
        
        # Prepare visualization data
        viz_data = {
            'convergence': {
                'iterations': list(range(len(results['fitness_history']))),
                'fitness': results['fitness_history']
            },
            'feature_importance': {
                'features': results['selected_feature_names'],
                'scores': results['feature_importance'].tolist()
            },
            'summary': {
                'n_selected': results['n_selected_features'],
                'best_fitness': results['best_fitness'],
                'optimization_time': results['optimization_time'],
                'n_iterations': results['n_iterations']
            }
        }
        
        return jsonify({
            'visualization_data': viz_data,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/export', methods=['POST'])
def export_results():
    """Export optimization results."""
    try:
        params = request.get_json()
        
        if not params:
            return jsonify({
                'error': 'No parameters provided',
                'status': 'error'
            }), 400
        
        cache_key = params.get('cache_key')
        if not cache_key or cache_key not in optimization_cache:
            return jsonify({
                'error': 'Optimization results not found. Please run optimization first.',
                'status': 'error'
            }), 400
        
        optimization_data = optimization_cache[cache_key]
        results = optimization_data['results']
        
        # Create export data
        export_data = {
            'metadata': {
                'timestamp': optimization_data['timestamp'],
                'n_particles': params.get('n_particles', 50),
                'n_iterations': params.get('n_iterations', 100),
                'classifier': params.get('classifier', 'mlp')
            },
            'results': {
                'selected_features': results['selected_features'],
                'selected_feature_names': results['selected_feature_names'],
                'best_fitness': results['best_fitness'],
                'n_selected_features': results['n_selected_features'],
                'optimization_time': results['optimization_time'],
                'n_iterations': results['n_iterations'],
                'fitness_history': results['fitness_history'],
                'feature_importance': results['feature_importance'].tolist()
            }
        }
        
        return jsonify({
            'export_data': export_data,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    # Run the API server
    logger.info("Starting PSO Feature Selection API...")
    app.run(host='0.0.0.0', port=5000, debug=True)