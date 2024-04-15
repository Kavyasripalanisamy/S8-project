import numpy as np
import pandas as pd
from hmmlearn import hmm
import joblib

data = pd.read_csv('gaitdataset.csv')
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

n_components = 2
covariance_type = "full"
n_iter = 100
model = hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter)
model.fit(X)

model_filename = 'hmm_model.joblib'
joblib.dump(model, model_filename)
print(f"Trained model saved as '{model_filename}'.")

print("\nModel Parameters:")
print(f"Number of Components: {model.n_components}")
print(f"Covariance Type: {model.covariance_type}")
print(f"Number of Iterations: {model.n_iter}")

accuracyscore = model.score(X)
print(f"\nAccuracy of the Training Data: {accuracyscore}")

hidden_states_train = model.predict(X)
print("\nHidden States for Training Data:")
print(hidden_states_train)

sample_probabilities_train = model.predict_proba(X)
print("\nSample Probabilities for Training Data:")
print(sample_probabilities_train)

new_input = np.array([[1.68, 70, 2.5, 33, 18, 10, 10.98, 1.204, 0]])
predicted_label = model.predict(new_input)
print("\nPredicted Label for New Input:", predicted_label[0])

predicted_probabilities = model.predict_proba(new_input)
print("\nPredicted Probabilities for New Input:")
print(predicted_probabilities)

print("\nModel Parameters Summary:")
print("Means:")
print(model.means_)
print("\nCovariances:")
print(model.covars_)
print("\nTransition Probabilities:")
print(model.transmat_)
