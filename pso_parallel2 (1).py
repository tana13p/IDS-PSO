import pandas as pd
import numpy as np
import random
import multiprocessing as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import time
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ------------------ DATA PREPROCESSING ------------------
def load_and_preprocess_data():
    df = pd.read_csv('KDDTrain+.txt', header=None, nrows=10000)
    df_test = pd.read_csv('KDDTest+.txt', header=None, nrows=10000)
    feature_names = [f"f{i}" for i in range(41)]
    columns = feature_names + ["label", "difficulty"]
    
    df.columns = columns
    df_test.columns = columns
    
    df = df.drop("difficulty", axis=1)
    df_test = df_test.drop("difficulty", axis=1)
    df.columns = list(range(42))
    df_test.columns = list(range(42))

    categorical_cols = [1, 2, 3]
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        df_test[col] = encoder.transform(df_test[col])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    y = y.apply(lambda x: 0 if x == 'normal' else 1)
    y_test = y_test.apply(lambda x: 0 if x == 'normal' else 1)

    # Feature selection using Random Forest (top 10 features)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]

    X_selected = X.iloc[:, indices]
    X_test_selected = X_test.iloc[:, indices]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    return X_scaled, X_test_scaled, y, y_test, indices

# ------------------ FITNESS FUNCTION ------------------
def fitness_function(params, X_train, y_train, X_val, y_val, all_features):
    selected_features = [i for i in range(len(all_features)) if params[i] >= 0.5]
    if len(selected_features) == 0:
        return 0  # Prevent empty selection

    X_train_selected = X_train[:, selected_features]
    X_val_selected = X_val[:, selected_features]

    clf = MLPClassifier(hidden_layer_sizes=(50,), alpha=0.001, max_iter=300, random_state=42)
    clf.fit(X_train_selected, y_train)
    preds = clf.predict(X_val_selected)
    
    return accuracy_score(y_val, preds)

# ------------------ SERIAL PSO ------------------
def serial_pso(X_train, y_train, X_val, y_val, all_features, num_particles=20, num_iterations=30):
    c1, c2, w = 0.5, 0.3, 0.9
    dim = len(all_features)

    particles = [np.array([random.choice([0, 1]) for _ in range(len(all_features))]) for _ in range(num_particles)]
    velocities = [np.zeros(dim) for _ in range(num_particles)]

    pbest = particles.copy()
    pbest_scores = [fitness_function(p, X_train, y_train, X_val, y_val, all_features) for p in particles]
    gbest = pbest[np.argmax(pbest_scores)]

    for iteration in range(num_iterations):
        for i in range(num_particles):
            r1, r2 = random.random(), random.random()
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest[i] - particles[i]) +
                             c2 * r2 * (gbest - particles[i]))
            particles[i] = particles[i] + velocities[i]

            particles[i] = np.clip(particles[i], 0, 1)

            score = fitness_function(particles[i], X_train, y_train, X_val, y_val, all_features)
            if score > pbest_scores[i]:
                pbest[i] = particles[i]
                pbest_scores[i] = score

        gbest = pbest[np.argmax(pbest_scores)]
        print(f"Iteration {iteration+1}/{num_iterations}, Best Accuracy: {max(pbest_scores):.4f}")

    return gbest


# ------------------ PARALLEL FITNESS ------------------
def parallel_fitness(particles, X_train, y_train, X_val, y_val, all_features):
    args_list = []
    for p in particles:
        selected_features = [i for i in range(len(all_features)) if p[i] >= 0.5]
        if not selected_features:
            selected_features = list(range(len(all_features)))
        
        X_train_selected = X_train[:, selected_features]
        X_val_selected = X_val[:, selected_features]
        args_list.append((X_train_selected, y_train, X_val_selected, y_val))
    
    with mp.Pool(mp.cpu_count()) as pool:
        output = pool.starmap(simple_fitness_function, args_list)
    
    return output

def simple_fitness_function(X_train_selected, y_train, X_val_selected, y_val):
    clf = MLPClassifier(hidden_layer_sizes=(50,), alpha=0.001, max_iter=300, random_state=42)
    clf.fit(X_train_selected, y_train)
    preds = clf.predict(X_val_selected)
    return accuracy_score(y_val, preds)

# ------------------ PARALLEL PSO ------------------
def parallel_pso(X_train, y_train, X_val, y_val, all_features, num_particles=20, num_iterations=30):
    c1, c2, w = 0.5, 0.3, 0.9
    dim = len(all_features)
    
    particles = np.random.choice([0, 1], size=(num_particles, len(all_features)))
    velocities = [np.zeros(dim) for _ in range(num_particles)]

    X_train = np.ascontiguousarray(X_train)
    y_train = np.ascontiguousarray(y_train)
    X_val = np.ascontiguousarray(X_val)
    y_val = np.ascontiguousarray(y_val)


    pbest = particles.copy(order='C')
    pbest_scores = parallel_fitness(particles, X_train, y_train, X_val, y_val, all_features)
    gbest = pbest[np.argmax(pbest_scores)]

    for iteration in range(num_iterations):
        for i in range(num_particles):
            r1, r2 = random.random(), random.random()
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest[i] - particles[i]) +
                             c2 * r2 * (gbest - particles[i]))
            particles[i] = particles[i] + velocities[i]

            particles[i] = np.clip(particles[i], 0, 1)

        scores = parallel_fitness(particles, X_train, y_train, X_val, y_val, all_features)
        for i, score in enumerate(scores):
            if score > pbest_scores[i]:
                pbest[i] = particles[i]
                pbest_scores[i] = score

        gbest = pbest[np.argmax(pbest_scores)]
        print(f"Iteration {iteration+1}/{num_iterations}, Best Accuracy: {max(pbest_scores):.4f}")

    return gbest

# ------------------ MAIN ------------------
X_train, X_test, y_train, y_test, feature_indices = load_and_preprocess_data()

# Serial PSO Optimization (Time Measurement)
print("Starting Serial PSO Optimization...")
start_time_serial = time.time()
serial_gbest = serial_pso(X_train, y_train, X_test, y_test, feature_indices)
serial_time = time.time() - start_time_serial
print(f"Time taken for Serial PSO: {serial_time:.4f} seconds")

# Parallel PSO Optimization (Time Measurement)
print("Starting Parallel PSO Optimization...")
start_time_parallel = time.time()
parallel_gbest = parallel_pso(X_train, y_train, X_test, y_test, feature_indices)
parallel_time = time.time() - start_time_parallel
print(f"Time taken for Parallel PSO: {parallel_time:.4f} seconds")

# Speedup Calculation
speedup = serial_time / parallel_time
print(f"Speedup using Parallel PSO: {speedup:.2f}x")

# ------------------ FEATURE SELECTION ------------------
# Serial PSO - Best Features
selected_features_serial = [i for i in range(len(serial_gbest)) if serial_gbest[i] >= 0.5]
X_train_selected_serial = X_train[:, selected_features_serial]
X_test_selected_serial = X_test[:, selected_features_serial]
print(f"Best Features Selected by Serial PSO: {selected_features_serial}")

# Final Model using Serial PSO results
final_model_serial = MLPClassifier(hidden_layer_sizes=(50,), alpha=0.001, max_iter=500, random_state=42)
final_model_serial.fit(X_train_selected_serial, y_train)
final_preds_serial = final_model_serial.predict(X_test_selected_serial)
print(f"Accuracy using Serial PSO: {accuracy_score(y_test, final_preds_serial)}")
print("Classification Report using Serial PSO:")
print(classification_report(y_test, final_preds_serial))

# ------------------ PARALLEL PSO - Best Features ------------------
# Parallel PSO - Best Features
selected_features_parallel = [feature_indices[i] for i in range(len(parallel_gbest)) if parallel_gbest[i] == 1]
print(f"Best Features Selected by Parallel PSO: {selected_features_parallel}")

# Final Model using Parallel PSO results
final_model_parallel = MLPClassifier(hidden_layer_sizes=(50,), alpha=0.001, max_iter=500, random_state=42)
final_model_parallel.fit(X_train[:, parallel_gbest[:len(feature_indices)]], y_train)
final_preds_parallel = final_model_parallel.predict(X_test[:, parallel_gbest[:len(feature_indices)]])
print(f"Accuracy using Parallel PSO: {accuracy_score(y_test, final_preds_parallel)}")
print("Classification Report using Parallel PSO:")
print(classification_report(y_test, final_preds_parallel))
