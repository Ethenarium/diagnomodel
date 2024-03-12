import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib
import time

# Load the dataset
df = pd.read_csv('patient_data.csv')

# Separate features and target variable
X = df.drop(['Diagnosis Result'], axis=1)
y = df['Diagnosis Result']

# Initialize logistic regression model
model = LogisticRegression(max_iter=1000, multi_class='ovr')

# Define hyperparameters for grid search
param_grid = {
    'C': [0.001],
    'class_weight': ['balanced', None]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=5), scoring='f1_macro', verbose=2)

start_time = time.time()
# Perform grid search to find the best hyperparameters
grid_search.fit(X, y)
end_time = time.time()

# Get the best model
best_model = grid_search.best_estimator_

# Save the best model
joblib.dump(best_model, 'best_logistic_regression_model.pkl')

print("Best hyperparameters found:", grid_search.best_params_)
print("Time taken for grid search:", end_time - start_time, "seconds")
