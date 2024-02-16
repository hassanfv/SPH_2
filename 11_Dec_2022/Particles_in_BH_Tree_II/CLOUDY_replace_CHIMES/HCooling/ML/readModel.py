

from tensorflow.keras.models import load_model

# Load the model
loaded_model = load_model('saved_model.pb')

# Assuming X_new is a new data point or array of data points with shape (M, 4)
# where M is the number of new samples you want to predict

# Use the loaded model to make predictions
#predictions = loaded_model.predict(X_new)

# Now, predictions contains the predicted values for the new data
#print(predictions)
