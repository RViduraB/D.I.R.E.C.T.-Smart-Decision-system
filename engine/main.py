
"""
First things first, we are importing all the essential packages we need. 
We've got 'os' for file paths, 'numpy' and 'pandas' for handling data like a pro, 
'joblib' to save our trained models, and 'sklearn' helpers to split and scale our dataset.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==========================================
# 1. NEURAL NETWORK COMPONENTS
# ==========================================

"""
These are our error-checking math tools. 
'mse' calculates the Mean Squared Error to see how far off our predictions are from reality, 
and 'mse_prime' gives us the gradient (direction) so we know exactly how to adjust things during training.
"""
def mse(y_true, y_pred): return np.mean(np.power(y_true - y_pred, 2))
def mse_prime(y_true, y_pred): return 2 * (y_pred - y_true)

"""
This is a Fully Connected (Dense) Layer class. 
When it kicks off, it initializes random weights and sets biases to zero. 
The 'forward' function calculates the layer's output, while the 'backward' function figures out 
the errors, passes them back, and updates the weights and biases using our learning rate.
"""
class DenseLayer:
    def __init__(self, input_size, neuron_count):
        self.weights = np.random.randn(input_size, neuron_count) * np.sqrt(1. / input_size)
        self.bias = np.zeros((1, neuron_count))
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        return input_error

"""
This is the ReLU activation layer. It acts like a simple filter! 
In the forward pass, if a number is positive, it passes right through; if it's negative, it gets turned into zero. 
In the backward pass, it only passes the gradient back if the original input was greater than zero.
"""
class ReLU:
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)
    def backward(self, output_error, learning_rate):
        return output_error * (self.input > 0)

"""
This is the big controller—the Neural Network manager. 
It lets us stack layers together, run data through them to make predictions, 
and train the whole model over multiple rounds (epochs) using backpropagation. 
It also saves the final trained weights into a file so we can reuse them later.
"""
class NeuralNetwork:
    def __init__(self): self.layers = []
    def add_layer(self, layer): self.layers.append(layer)
    def predict(self, input_data):
        result = input_data
        for layer in self.layers: result = layer.forward(result)
        return result
    def train(self, x_train, y_train, epochs, learning_rate):
        print(f"\n[AI] Training Initiated... Samples: {len(x_train)}")
        for epoch in range(epochs):
            display_error = 0
            for i in range(len(x_train)):
                output = x_train[i:i+1]
                for layer in self.layers: output = layer.forward(output)
                display_error += mse(y_train[i:i+1], output)
                err = mse_prime(y_train[i:i+1], output)
                for layer in reversed(self.layers): err = layer.backward(err, learning_rate)
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {display_error/len(x_train):.6f}")

    def save_weights(self, folder):
        os.makedirs(folder, exist_ok=True)
        # Saving weights and biases as a list
        weights_data = [{'weights': l.weights, 'bias': l.bias} for l in self.layers if isinstance(l, DenseLayer)]
        joblib.dump(weights_data, os.path.join(folder, 'network_weights.pkl'))

# ==========================================
# 2. CORE EXECUTION LOGIC
# ==========================================

"""
This is the main project workflow function! 
It handles everything step-by-step: sets up folders, reads the dataset, encodes text columns (like District) into numbers, 
splits and scales features, builds and trains our network, and finally loops through the data to predict 
rice production and compare it against the country's demand to see if there's a surplus or deficit.
"""
def run_project_flow():
    # Setting up file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    data_path = os.path.join(base_dir, "data", "my_data.csv")
    
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return [], 0, 2400000

    # Reading the dataset
    df = pd.read_csv(data_path)
    
    # 1. Creating and saving encoders
    le_dist, le_seed = LabelEncoder(), LabelEncoder()
    df['District_Enc'] = le_dist.fit_transform(df['District'])
    df['Seed_Enc'] = le_seed.fit_transform(df['Seed_Variety'])
    
    joblib.dump(le_dist, os.path.join(models_dir, 'le_district.pkl'))
    joblib.dump(le_seed, os.path.join(models_dir, 'le_seed.pkl'))

    # 2. Features and Scaling
    feature_cols = ['Year', 'District_Enc', 'Land_Area', 'Seed_Enc', 'Rainfall_mm', 'Soil_pH', 'Pest_Damage']
    X = df[feature_cols].values
    y = df['Yield_MT'].values.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    
    X_full_scaled = scaler.transform(X)
    
    # 3. Model training
    net = NeuralNetwork()
    net.add_layer(DenseLayer(X_train_scaled.shape[1], 12))
    net.add_layer(ReLU())
    net.add_layer(DenseLayer(12, 1))
    
    net.train(X_train_scaled, y_train, epochs=500, learning_rate=0.01)
    net.save_weights(models_dir)

    # 4. Preparing results and the terminal table
    results_list = []
    total_rice_production = 0
    CONVERSION = 0.68
    DEMAND = 2400000

    print("\n" + "="*125)
    print(f"{'District':<15} | {'Exp.Paddy':<10} | {'Exp.Rice':<10} | {'Land':<8} | {'Prod.Paddy':<12} | {'Prod.Rice':<12}")
    print("-" * 125)

    for i, row in df.iterrows():
        scaled_input = X_full_scaled[i:i+1]
        exp_paddy_yield = max(0, float(net.predict(scaled_input)[0][0]))
        exp_rice_yield = exp_paddy_yield * CONVERSION
        
        prod_paddy = exp_paddy_yield * row['Land_Area']
        prod_rice = prod_paddy * CONVERSION
        total_rice_production += prod_rice
        
        # Printing to the terminal
        print(f"{row['District']:<15} | {exp_paddy_yield:>10.2f} | {exp_rice_yield:>10.2f} | {row['Land_Area']:>8.1f} | {prod_paddy:>12.2f} | {prod_rice:>12.2f}")

        results_list.append({
            'Year': int(row['Year']),
            'District': row['District'],
            'Exp_Paddy': round(exp_paddy_yield, 2),
            'Exp_Rice': round(exp_rice_yield, 2),
            'Land': float(row['Land_Area']),
            'Prod_Paddy': round(prod_paddy, 2),
            'Prod_Rice': round(prod_rice, 2)
        })

    gap = total_rice_production - DEMAND
    status = "SURPLUS (EXPORT)" if gap > 0 else "DEFICIT (IMPORT)"

    print("-" * 125)
    print(f"Total Rice Production: {total_rice_production:,.2f} MT | Country's Demand: {DEMAND:,.2f} MT")
    print(f"Conclusion: {status} | Quantity: {abs(gap):,.2f} MT")
    print("="*125 + "\n")

    return results_list, total_rice_production, DEMAND

"""
This is the standard execution entry point. 
If we run this specific script directly, it kicks off the whole pipeline and reports back when training is finished.
"""
if __name__ == "__main__":
    data, total, dem = run_project_flow()
    print(f"\nTraining Complete. Total Rice Production: {total:,.2f} MT")