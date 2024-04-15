import numpy as np
from hmmlearn import hmm
import joblib
model = joblib.load('hmm_model.joblib')
new_input = np.array([[65,1.72,72,0,0,0,0,1.542,1.532]])
predicted_label = model.predict(new_input)
if(predicted_label[0]==0):
    print("Predicted Label: Control Object")
else:
    print("Predicted Label: Parkinson Disease")