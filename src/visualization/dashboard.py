"""
Interactive dashboard for PSO feature selection.

This module provides a comprehensive dashboard for visualizing
the PSO optimization process and feature selection results.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pso_optimizer import PSOOptimizer, PSOConfig
from data.data_loader import DataLoader, load_and_preprocess_dataset
from core.feature_selector import FeatureSelector, FeatureSelectionEvaluator

logger = logging.getLogger(__name__)

class PSODashboard:
    """
    Interactive dashboard for PSO feature selection.
    """
    
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[
            'https://codepen.io/chriddyp/pen/bWLwgP.css'
        ])
        self.app.title = "PSO Feature Selection Dashboard"
        
        # Initialize components
        self.data_loader = DataLoader()
        self.pso_optimizer = None
        self.current_data = None
        self.optimization_results = None
        
        # Setup layout
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("PSO Feature Selection Dashboard", 
                       style={'textAlign': 'center', 'marginBottom': 30}),
                html.P("Interactive visualization of Particle Swarm Optimization for feature selection",
                      style={'textAlign': 'center', 'fontSize': 16})
            ]),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.H3("Dataset Selection"),
                    dcc.Dropdown(
                        id='dataset-dropdown',
                        options=[
                            {'label': 'Wine', 'value': 'wine'},
                            {'label': 'Breast Cancer', 'value': 'breast_cancer'},
                            {'label': 'Iris', 'value': 'iris'},
                            {'label': 'Synthetic', 'value': 'synthetic'},
                            {'label': 'KDD Cup', 'value': 'kdd_cup'}
                        ],
                        value='wine',
                        style={'marginBottom': 20}
                    ),
                    html.Button('Load Dataset', id='load-dataset-btn', 
                              style={'marginBottom': 20})
                ], className='six columns'),
                
                html.Div([
                    html.H3("PSO Configuration"),
                    html.Label("Number of Particles:"),
                    dcc.Input(id='n-particles', type='number', value=50, min=10, max=200),
                    html.Br(),
                    html.Label("Number of Iterations:"),
                    dcc.Input(id='n-iterations', type='number', value=100, min=10, max=500),
                    html.Br(),
                    html.Label("Classifier:"),
                    dcc.Dropdown(
                        id='classifier-dropdown',
                        options=[
                            {'label': 'MLP', 'value': 'mlp'},
                            {'label': 'Random Forest', 'value': 'rf'},
                            {'label': 'SVM', 'value': 'svm'},
                            {'label': 'Logistic Regression', 'value': 'lr'}
                        ],
                        value='mlp'
                    ),
                    html.Br(),
                    html.Button('Start Optimization', id='start-optimization-btn',
                              style={'marginTop': 10})
                ], className='six columns')
            ], className='row', style={'marginBottom': 30}),
            
            # Status and Progress
            html.Div([
                html.Div(id='status-display', style={'textAlign': 'center', 'fontSize': 16}),
                dcc.Interval(id='progress-interval', interval=1000, n_intervals=0, disabled=True)
            ]),
            
            # Main Content
            html.Div([
                # Convergence Plot
                html.Div([
                    html.H3("Optimization Progress"),
                    dcc.Graph(id='convergence-plot')
                ], className='six columns'),
                
                # Feature Importance
                html.Div([
                    html.H3("Feature Importance"),
                    dcc.Graph(id='feature-importance-plot')
                ], className='six columns')
            ], className='row', style={'marginTop': 30}),
            
            # Results Section
            html.Div([
                html.H3("Optimization Results"),
                html.Div(id='results-display')
            ], style={'marginTop': 30}),
            
            # Comparison Section
            html.Div([
                html.H3("Method Comparison"),
                html.Button('Compare Methods', id='compare-methods-btn'),
                html.Div(id='comparison-results')
            ], style={'marginTop': 30})
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('status-display', 'children'),
             Output('progress-interval', 'disabled')],
            [Input('load-dataset-btn', 'n_clicks'),
             Input('start-optimization-btn', 'n_clicks')],
            [State('dataset-dropdown', 'value')]
        )
        def update_status(load_clicks, start_clicks, dataset):
            ctx = callback_context
            if not ctx.triggered:
                return "Ready to load dataset", True
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == 'load-dataset-btn':
                try:
                    self.current_data = load_and_preprocess_dataset(dataset)
                    return f"Dataset '{dataset}' loaded successfully! Shape: {self.current_data['X_train'].shape}", True
                except Exception as e:
                    return f"Error loading dataset: {str(e)}", True
            
            elif trigger_id == 'start-optimization-btn':
                if self.current_data is None:
                    return "Please load a dataset first!", True
                return "Optimization in progress...", False
            
            return "Ready", True
        
        @self.app.callback(
            Output('convergence-plot', 'figure'),
            [Input('progress-interval', 'n_intervals')]
        )
        def update_convergence_plot(n_intervals):
            if self.pso_optimizer is None or not hasattr(self.pso_optimizer, 'fitness_history'):
                return go.Figure()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=self.pso_optimizer.fitness_history,
                mode='lines+markers',
                name='Best Fitness',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title='PSO Convergence',
                xaxis_title='Iteration',
                yaxis_title='Fitness Score',
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output('feature-importance-plot', 'figure'),
            [Input('progress-interval', 'n_intervals')]
        )
        def update_feature_importance_plot(n_intervals):
            if self.pso_optimizer is None or not hasattr(self.pso_optimizer, 'feature_importance'):
                return go.Figure()
            
            if self.current_data is None:
                return go.Figure()
            
            feature_names = self.current_data['feature_names']
            importance = self.pso_optimizer.feature_importance
            
            # Get top 20 features
            top_indices = np.argsort(importance)[-20:]
            top_names = [feature_names[i] for i in top_indices]
            top_scores = importance[top_indices]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_scores,
                y=top_names,
                orientation='h',
                marker=dict(color='lightblue')
            ))
            
            fig.update_layout(
                title='Top 20 Feature Importance',
                xaxis_title='Importance Score',
                yaxis_title='Features',
                height=600
            )
            
            return fig
        
        @self.app.callback(
            Output('results-display', 'children'),
            [Input('start-optimization-btn', 'n_clicks')],
            [State('n-particles', 'value'),
             State('n-iterations', 'value'),
             State('classifier-dropdown', 'value')]
        )
        def run_optimization(n_clicks, n_particles, n_iterations, classifier):
            if n_clicks is None or self.current_data is None:
                return ""
            
            try:
                # Configure PSO
                config = PSOConfig(
                    n_particles=n_particles,
                    n_iterations=n_iterations,
                    classifier=classifier,
                    parallel=True
                )
                
                # Initialize optimizer
                self.pso_optimizer = PSOOptimizer(config)
                
                # Run optimization
                X_train = self.current_data['X_train']
                y_train = self.current_data['y_train']
                feature_names = self.current_data['feature_names']
                
                results = self.pso_optimizer.optimize(X_train, y_train, feature_names)
                self.optimization_results = results
                
                # Display results
                return html.Div([
                    html.H4("Optimization Complete!"),
                    html.P(f"Selected Features: {len(results['selected_features'])}"),
                    html.P(f"Best Fitness: {results['best_fitness']:.4f}"),
                    html.P(f"Optimization Time: {results['optimization_time']:.2f} seconds"),
                    html.P(f"Iterations: {results['n_iterations']}"),
                    html.H5("Selected Features:"),
                    html.Ul([html.Li(feat) for feat in results['selected_feature_names'][:10]]),
                    html.P("..." if len(results['selected_feature_names']) > 10 else "")
                ])
                
            except Exception as e:
                return html.Div([
                    html.H4("Optimization Error"),
                    html.P(f"Error: {str(e)}")
                ])
        
        @self.app.callback(
            Output('comparison-results', 'children'),
            [Input('compare-methods-btn', 'n_clicks')]
        )
        def compare_methods(n_clicks):
            if n_clicks is None or self.current_data is None:
                return ""
            
            try:
                X_train = self.current_data['X_train']
                y_train = self.current_data['y_train']
                
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
                
                # Create comparison table
                table = html.Table([
                    html.Thead([
                        html.Tr([html.Th(col) for col in comparison_df.columns])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td(comparison_df.iloc[i][col]) 
                            for col in comparison_df.columns
                        ]) for i in range(len(comparison_df))
                    ])
                ])
                
                return html.Div([
                    html.H4("Method Comparison Results"),
                    table
                ])
                
            except Exception as e:
                return html.Div([
                    html.H4("Comparison Error"),
                    html.P(f"Error: {str(e)}")
                ])
    
    def run(self, debug=True, port=8050):
        """Run the dashboard."""
        logger.info(f"Starting dashboard on port {port}")
        self.app.run_server(debug=debug, port=port)

def create_standalone_plots(optimization_results: Dict[str, Any], 
                          feature_names: List[str],
                          save_dir: str = "results/visualizations"):
    """Create standalone plots for the optimization results."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Convergence plot
    plt.figure(figsize=(10, 6))
    plt.plot(optimization_results['fitness_history'])
    plt.title('PSO Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance plot
    importance = optimization_results['feature_importance']
    top_indices = np.argsort(importance)[-20:]
    top_names = [feature_names[i] for i in top_indices]
    top_scores = importance[top_indices]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_scores, y=top_names)
    plt.title('Top 20 Feature Importance Scores')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {save_dir}")

# Example usage
if __name__ == "__main__":
    # Create and run dashboard
    dashboard = PSODashboard()
    dashboard.run(debug=True, port=8050)