
import joblib
import numpy as np
import os

class Predictor:
    """
    Custom Multi-Layer Perceptron (MLP) inference class.
    Loads pre-trained neural network weights and biases from a serialized file 
    to perform manual forward propagation using NumPy.
    """

    def __init__(self, weights_path):
        """
        Initializes the Predictor instance by loading weights and biases.

        Parameters:
        weights_path (str): The absolute or relative file path to 'network_weights.pkl'.
        """
        # Load saved weights and biases
        self.layers_data = joblib.load(weights_path)

    def relu(self, x):
        """
        Rectified Linear Unit (ReLU) activation function.
        Replaces all negative numerical values in the matrix with zero.

        Parameters:
        x (numpy.ndarray): Input matrix or layer activations.

        Returns:
        numpy.ndarray: Matrix with activated values, where f(x) = max(0, x).
        """
        return np.maximum(0, x)

    def predict(self, x):
        """
        Executes the forward propagation process through all layers.
        Performs matrix multiplication (dot product) between input and layer weights,
        adds the bias vector, and applies the ReLU activation function for all hidden layers.

        Parameters:
        x (numpy.ndarray): Standardized/Scaled feature matrix of shape (1, 7).

        Returns:
        numpy.ndarray: The raw prediction value output from the final output layer.
        """
        # x is a matrix of shape (1, 7)
        result = x
        for i, layer in enumerate(self.layers_data):
            # Matrix multiplication (1,7) x (7,12) -> (1,12)
            result = np.dot(result, layer['weights']) + layer['bias']
            
            # Apply ReLU activation only if it is not the final output layer
            if i < len(self.layers_data) - 1:
                result = self.relu(result)
        return result

def get_prediction(year, district, land, seed, rain, ph, pest):
    """
    Main pipeline function that prepares features and generates an end-to-end prediction.
    It loads transformers, encodes text strings to numbers, scales numerical inputs,
    and passes the data to the Predictor class.

    Parameters:
    year (int): The cultivation year.
    district (str): Categorical text name of the district (e.g., "Ampara").
    land (float): Land area size.
    seed (str): Categorical text name of the seed type.
    rain (float): Average rainfall.
    ph (float): Soil pH value.
    pest (float): Pesticide/pests impact index.

    Returns:
    float: Final expected prediction value, bounded to a minimum of 0.0.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')

    # 1. Load saved models and transformers
    le_dist = joblib.load(os.path.join(models_dir, 'le_district.pkl'))
    le_seed = joblib.load(os.path.join(models_dir, 'le_seed.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    weights_path = os.path.join(models_dir, 'network_weights.pkl')

    # 2. Encoding (Convert categorical string data to numerical values)
    dist_enc = le_dist.transform([district])[0]
    seed_enc = le_seed.transform([seed])[0]
    
    # Format features matrix to shape (1, 7)
    features = np.array([[year, dist_enc, land, seed_enc, rain, ph, pest]], dtype=float)
    
    # Scaling (Scaler ensures that all 7 required features are present)
    features_scaled = scaler.transform(features)

    # 3. Prediction
    model = Predictor(weights_path)
    prediction = model.predict(features_scaled)
    
    # Extract the final result as a single scalar value
    return max(0, float(prediction[0][0]))

if __name__ == "__main__":
    """
    Main entry point for local execution and quick integration testing.
    Feeds sample values into the pipeline and prints the final outcome while catching errors.
    """
    # Test run
    try:
        result = get_prediction(2027, "Ampara", 500.0, "Bg352", 1200.5, 6.5, 2.0)
        print(f"\n[AI Prediction] Expected Yield: {result:.2f} MT")
    except Exception as e:
        print(f"Error: {e}")

    
