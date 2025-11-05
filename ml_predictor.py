import pandas as pd
import numpy as np
import requests
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Placeholder for the model file path
MODEL_PATH = 'ndvi_yield_predictor/random_forest_model.pkl'

class NDVIVariablePredictor:
    def __init__(self, model=None):
        self.model = model
        self.features = ['EVI', 'LST', 'Soil_Moisture', 'Rice_Option', 'Maize_Option']
        self.target = 'Yield'
        self.scaler = None # Placeholder for a scaler if needed

    def _synthesize_data_from_coords(self, lat, lon, crop):
        """
        Synthesizes or fetches data for prediction based on coordinates and crop.
        
        NOTE: This function is a placeholder for a real API call to a service like VEDAS.
        You need to replace the placeholder logic with your actual VEDAS API call.
        
        The VEDAS API call should fetch the following data points for the given
        latitude, longitude, and time period:
        - EVI (Enhanced Vegetation Index)
        - LST (Land Surface Temperature)
        - Soil_Moisture
        
        The 'crop' parameter is used to set the 'Rice_Option' and 'Maize_Option' features.
        
        Parameters:
        - lat (float): Latitude of the location.
        - lon (float): Longitude of the location.
        - crop (str): The crop type ('Rice' or 'Maize').
        
        Returns:
        - pd.DataFrame: A DataFrame with a single row containing the features
                        ['EVI', 'LST', 'Soil_Moisture', 'Rice_Option', 'Maize_Option'].
        """
        
        # --- START VEDAS API INTEGRATION PLACEHOLDER ---
        
        # Replace these with your actual VEDAS API key and endpoint
        VEDAS_API_KEY = "YOUR_VEDAS_API_KEY"
        VEDAS_ENDPOINT = "YOUR_VEDAS_API_ENDPOINT"
        
        # Example API call structure (you will need to adjust parameters)
        # api_url = f"{VEDAS_ENDPOINT}/data?lat={lat}&lon={lon}&key={VEDAS_API_KEY}"
        
        # try:
        #     response = requests.get(api_url)
        #     response.raise_for_status() # Raise an exception for bad status codes
        #     data = response.json()
            
            # Assuming the API returns a dictionary with 'evi', 'lst', 'soil_moisture'
            # evi = data.get('evi')
            # lst = data.get('lst')
            # soil_moisture = data.get('soil_moisture')
            
        # except requests.exceptions.RequestException as e:
        #     print(f"Error fetching data from VEDAS API: {e}")
            # Fallback to a default or raise an error
            # evi, lst, soil_moisture = 0.5, 300.0, 0.3 
            
        # --- SIMULATION/MOCK DATA (REMOVE FOR REAL API) ---
        # For demonstration, we use mock data. Replace this with the data from your API call.
        np.random.seed(42)
        evi = np.random.uniform(0.3, 0.8)
        lst = np.random.uniform(290, 310)
        soil_moisture = np.random.uniform(0.2, 0.5)
        # --- END SIMULATION/MOCK DATA ---
        
        # --- END VEDAS API INTEGRATION PLACEHOLDER ---

        rice_option = 1 if crop.lower() == 'rice' else 0
        maize_option = 1 if crop.lower() == 'maize' else 0

        data = {
            'EVI': [evi],
            'LST': [lst],
            'Soil_Moisture': [soil_moisture],
            'Rice_Option': [rice_option],
            'Maize_Option': [maize_option]
        }
        
        return pd.DataFrame(data)

    def train_model(self, data_path='data/crop_data.csv'):
        """Trains a Random Forest Regressor model."""
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}. Cannot train model.")
            return

        X = df[self.features]
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate (optional)
        y_pred = self.model.predict(X_test)
        print(f"Model trained. RMSE on test set: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

        # In a real scenario, you would save the model here
        # import joblib
        # joblib.dump(self.model, MODEL_PATH)
        
    def predict_single_yield(self, lat, lon, crop):
        """Predicts the yield for a single location and crop."""
        if self.model is None:
            print("Model not trained. Attempting to train with default data.")
            self.train_model()
            if self.model is None:
                return "Error: Model could not be trained or loaded."

        # 1. Get the feature data (from API or mock)
        input_df = self._synthesize_data_from_coords(lat, lon, crop)
        
        # 2. Predict
        prediction = self.model.predict(input_df[self.features])[0]
        
        # 3. Return result and the input data used for prediction
        input_data_str = input_df.iloc[0].to_json()
        return f"{prediction:.2f}", input_data_str

# Helper function to create a mock data file for demonstration
def create_mock_data(path='data/crop_data.csv'):
    """Creates a mock CSV file for model training."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.random.seed(42)
    
    # Generate mock data
    n_samples = 100
    evi = np.random.uniform(0.3, 0.8, n_samples)
    lst = np.random.uniform(290, 310, n_samples)
    soil_moisture = np.random.uniform(0.2, 0.5, n_samples)
    
    # Crop options: 50% Rice, 50% Maize
    crop_options = np.random.choice(['Rice', 'Maize'], n_samples)
    rice_option = (crop_options == 'Rice').astype(int)
    maize_option = (crop_options == 'Maize').astype(int)
    
    # Simple yield model: Yield is a function of EVI, Soil Moisture, and crop type
    # Rice generally has higher yield than Maize in this mock data
    base_yield = 50 + 100 * evi + 50 * soil_moisture
    yield_noise = np.random.normal(0, 5, n_samples)
    
    yield_data = base_yield + 10 * rice_option - 5 * maize_option + yield_noise
    yield_data = np.clip(yield_data, 50, 200) # Clip to realistic range
    
    data = pd.DataFrame({
        'EVI': evi,
        'LST': lst,
        'Soil_Moisture': soil_moisture,
        'Rice_Option': rice_option,
        'Maize_Option': maize_option,
        'Yield': yield_data
    })
    
    data.to_csv(path, index=False)
    print(f"Mock data created at {path}")

if __name__ == '__main__':
    # Example usage:
    create_mock_data()
    predictor = NDVIVariablePredictor()
    
    # Train the model
    predictor.train_model()
    
    # Predict for a location (e.g., a good rice-growing area)
    lat_rice, lon_rice = 20.0, 80.0
    yield_rice, data_rice = predictor.predict_single_yield(lat_rice, lon_rice, 'Rice')
    print(f"\nPredicted Rice Yield at ({lat_rice}, {lon_rice}): {yield_rice}")
    print(f"Input Data Used: {data_rice}")

    # Predict for a location (e.g., a good maize-growing area)
    lat_maize, lon_maize = 25.0, 75.0
    yield_maize, data_maize = predictor.predict_single_yield(lat_maize, lon_maize, 'Maize')
    print(f"\nPredicted Maize Yield at ({lat_maize}, {lon_maize}): {yield_maize}")
    print(f"Input Data Used: {data_maize}")
