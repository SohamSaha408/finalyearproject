from flask import Flask, render_template, request
from ml_predictor import NDVIVariablePredictor, create_mock_data
import json

app = Flask(__name__, template_folder='templates')

# Initialize the predictor and train the model on startup
# This is a simple approach; for production, the model should be loaded from a file.
try:
    create_mock_data(path='data/crop_data.csv')
    predictor = NDVIVariablePredictor()
    predictor.train_model(data_path='data/crop_data.csv')
except Exception as e:
    print(f"Error during model initialization: {e}")
    predictor = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    input_data_used = None
    error_message = None

    if request.method == 'POST':
        if predictor is None:
            error_message = "Model is not initialized. Check server logs for errors."
        else:
            try:
                lat = float(request.form.get('latitude'))
                lon = float(request.form.get('longitude'))
                crop = request.form.get('crop')

                # Call the prediction function
                prediction, input_data_json = predictor.predict_single_yield(lat, lon, crop)
                
                prediction_result = prediction
                
                # Parse the JSON string back to a dict for display
                input_data_used = json.loads(input_data_json)

            except ValueError:
                error_message = "Invalid input. Please ensure Latitude and Longitude are valid numbers."
            except Exception as e:
                error_message = f"An error occurred during prediction: {e}"

    return render_template('index.html', 
                           prediction_result=prediction_result, 
                           input_data_used=input_data_used,
                           error_message=error_message)

if __name__ == '__main__':
    # The mock data is created and the model is trained when the app is initialized
    app.run(debug=True)
