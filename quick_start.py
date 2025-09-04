#!/usr/bin/env python3
"""
Quick start script for Intrusion Detection System (IDS) project.

This script provides a simple way to get started with the IDS project
and demonstrates its capabilities for network intrusion detection.
"""

import os
import sys
import subprocess
import time

def print_banner():
    """Print project banner."""
    print("=" * 60)
    print("🛡️ Intrusion Detection System (IDS) - Quick Start")
    print("=" * 60)
    print("AI-Powered Network Intrusion Detection with PSO Feature Selection")
    print("=" * 60)

def check_requirements():
    """Check if requirements are installed."""
    print("\n📋 Checking requirements...")
    
    try:
        import numpy
        import pandas
        import sklearn
        import matplotlib
        import seaborn
        import plotly
        import dash
        import flask
        print("✅ All required packages are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    directories = [
        'results',
        'results/visualizations',
        'results/experiments',
        'data/raw',
        'data/processed'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")

def run_ids_example():
    """Run the IDS example."""
    print("\n🔬 Running IDS PSO optimization example...")
    
    try:
        # Import and run IDS example
        sys.path.append('src')
        from example_usage import main
        main()
        print("✅ IDS example completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error running IDS example: {e}")
        return False

def start_ids_dashboard():
    """Start the IDS interactive dashboard."""
    print("\n📊 Starting IDS interactive dashboard...")
    print("IDS Dashboard will be available at: http://localhost:8050")
    print("Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([sys.executable, '-m', 'src.visualization.dashboard'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 IDS Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting IDS dashboard: {e}")

def start_ids_api():
    """Start the IDS REST API."""
    print("\n🌐 Starting IDS REST API...")
    print("IDS API will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the API")
    
    try:
        subprocess.run([sys.executable, '-m', 'src.api.app'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 IDS API stopped by user")
    except Exception as e:
        print(f"❌ Error starting IDS API: {e}")

def show_menu():
    """Show interactive menu."""
    while True:
        print("\n" + "=" * 40)
        print("🎯 IDS System - What would you like to do?")
        print("=" * 40)
        print("1. Run IDS PSO optimization example")
        print("2. Start IDS interactive dashboard")
        print("3. Start IDS REST API")
        print("4. Run IDS tests")
        print("5. View IDS documentation")
        print("6. Deploy with Docker")
        print("7. Exit")
        print("=" * 40)
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            run_ids_example()
        elif choice == '2':
            start_ids_dashboard()
        elif choice == '3':
            start_ids_api()
        elif choice == '4':
            run_tests()
        elif choice == '5':
            show_documentation()
        elif choice == '6':
            deploy_with_docker()
        elif choice == '7':
            print("\n👋 Thanks for using the IDS System!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def run_tests():
    """Run the IDS test suite."""
    print("\n🧪 Running IDS tests...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'], check=True)
        print("✅ All IDS tests passed!")
    except subprocess.CalledProcessError:
        print("❌ Some IDS tests failed. Check the output above.")
    except Exception as e:
        print(f"❌ Error running IDS tests: {e}")

def deploy_with_docker():
    """Deploy IDS system with Docker."""
    print("\n🐳 Deploying IDS system with Docker...")
    print("This will start the IDS API and Dashboard services")
    
    try:
        subprocess.run(['docker-compose', 'up', '--build'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Docker deployment stopped by user")
    except Exception as e:
        print(f"❌ Error deploying with Docker: {e}")
        print("Make sure Docker and docker-compose are installed")

def show_documentation():
    """Show IDS project documentation."""
    print("\n📚 IDS Project Documentation:")
    print("=" * 40)
    print("📖 README.md - Main IDS documentation")
    print("📊 PROJECT_SUMMARY.md - Comprehensive IDS summary")
    print("🔬 notebooks/ - IDS analysis notebooks")
    print("🧪 tests/ - IDS test suite")
    print("📁 src/ - IDS source code")
    print("=" * 40)
    print("\nKey files to explore:")
    print("- README.md: IDS getting started guide")
    print("- example_usage.py: IDS usage examples")
    print("- notebooks/01_data_exploration.ipynb: IDS data analysis")
    print("- src/core/pso_optimizer.py: IDS PSO implementation")
    print("- src/data/data_loader.py: KDD Cup dataset loader")

def main():
    """Main function."""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please install requirements first:")
        print("pip install -r requirements.txt")
        return
    
    # Setup directories
    setup_directories()
    
    # Show menu
    show_menu()

if __name__ == "__main__":
    main()