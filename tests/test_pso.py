"""
Tests for PSO optimizer.
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.pso_optimizer import PSOOptimizer, PSOConfig, Particle

class TestPSOOptimizer:
    """Test cases for PSO optimizer."""
    
    def test_pso_config(self):
        """Test PSO configuration."""
        config = PSOConfig()
        assert config.n_particles == 50
        assert config.n_iterations == 100
        assert config.w == 0.9
        assert config.c1 == 2.0
        assert config.c2 == 2.0
    
    def test_particle_initialization(self):
        """Test particle initialization."""
        config = PSOConfig()
        particle = Particle(10, config)
        
        assert len(particle.position) == 10
        assert len(particle.velocity) == 10
        assert particle.fitness == -np.inf
        assert particle.best_fitness == -np.inf
    
    def test_particle_velocity_update(self):
        """Test particle velocity update."""
        config = PSOConfig()
        particle = Particle(5, config)
        global_best = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Test velocity update
        particle.update_velocity(global_best, config)
        assert len(particle.velocity) == 5
    
    def test_particle_position_update(self):
        """Test particle position update."""
        config = PSOConfig()
        particle = Particle(5, config)
        
        # Test position update
        particle.update_position(config)
        assert len(particle.position) == 5
        assert np.all(particle.position >= 0)
        assert np.all(particle.position <= 1)
    
    def test_particle_feature_selection(self):
        """Test particle feature selection."""
        config = PSOConfig()
        particle = Particle(10, config)
        particle.position = np.array([0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6, 0.1, 0.5])
        
        selected = particle.get_selected_features(threshold=0.5)
        expected = [1, 3, 5, 6, 7, 9]  # Indices with values >= 0.5
        
        assert selected == expected
    
    def test_pso_optimizer_initialization(self):
        """Test PSO optimizer initialization."""
        config = PSOConfig(n_particles=20, n_iterations=50)
        optimizer = PSOOptimizer(config)
        
        assert optimizer.config.n_particles == 20
        assert optimizer.config.n_iterations == 50
        assert optimizer.global_best_fitness == -np.inf
    
    def test_pso_optimization_synthetic_data(self):
        """Test PSO optimization on synthetic data."""
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(20)]
        
        # Configure PSO
        config = PSOConfig(
            n_particles=10,
            n_iterations=20,
            parallel=False,  # Disable parallel for testing
            classifier='mlp'
        )
        
        # Initialize optimizer
        optimizer = PSOOptimizer(config)
        
        # Run optimization
        results = optimizer.optimize(X, y, feature_names)
        
        # Check results
        assert 'selected_features' in results
        assert 'best_fitness' in results
        assert 'n_selected_features' in results
        assert 'optimization_time' in results
        assert len(results['selected_features']) > 0
        assert results['best_fitness'] >= 0
        assert results['optimization_time'] > 0
    
    def test_pso_optimization_different_classifiers(self):
        """Test PSO optimization with different classifiers."""
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        feature_names = [f"feature_{i}" for i in range(10)]
        
        classifiers = ['mlp', 'rf', 'svm', 'lr']
        
        for classifier in classifiers:
            config = PSOConfig(
                n_particles=5,
                n_iterations=10,
                parallel=False,
                classifier=classifier
            )
            
            optimizer = PSOOptimizer(config)
            results = optimizer.optimize(X, y, feature_names)
            
            assert len(results['selected_features']) > 0
            assert results['best_fitness'] >= 0
    
    def test_pso_parallel_vs_serial(self):
        """Test PSO parallel vs serial execution."""
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        feature_names = [f"feature_{i}" for i in range(10)]
        
        # Serial PSO
        config_serial = PSOConfig(
            n_particles=10,
            n_iterations=20,
            parallel=False,
            classifier='mlp'
        )
        
        optimizer_serial = PSOOptimizer(config_serial)
        results_serial = optimizer_serial.optimize(X, y, feature_names)
        
        # Parallel PSO
        config_parallel = PSOConfig(
            n_particles=10,
            n_iterations=20,
            parallel=True,
            classifier='mlp'
        )
        
        optimizer_parallel = PSOOptimizer(config_parallel)
        results_parallel = optimizer_parallel.optimize(X, y, feature_names)
        
        # Both should produce valid results
        assert len(results_serial['selected_features']) > 0
        assert len(results_parallel['selected_features']) > 0
        assert results_serial['best_fitness'] >= 0
        assert results_parallel['best_fitness'] >= 0
    
    def test_pso_early_stopping(self):
        """Test PSO early stopping functionality."""
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        feature_names = [f"feature_{i}" for i in range(10)]
        
        config = PSOConfig(
            n_particles=5,
            n_iterations=100,
            early_stopping=True,
            patience=5,
            parallel=False,
            classifier='mlp'
        )
        
        optimizer = PSOOptimizer(config)
        results = optimizer.optimize(X, y, feature_names)
        
        # Should stop early due to no improvement
        assert results['n_iterations'] <= 100
        assert len(results['fitness_history']) == results['n_iterations']
    
    def test_pso_feature_constraints(self):
        """Test PSO with feature constraints."""
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        feature_names = [f"feature_{i}" for i in range(10)]
        
        config = PSOConfig(
            n_particles=5,
            n_iterations=10,
            min_features=2,
            max_features=5,
            parallel=False,
            classifier='mlp'
        )
        
        optimizer = PSOOptimizer(config)
        results = optimizer.optimize(X, y, feature_names)
        
        # Check feature constraints
        n_selected = len(results['selected_features'])
        assert n_selected >= config.min_features
        assert n_selected <= config.max_features

if __name__ == "__main__":
    pytest.main([__file__])