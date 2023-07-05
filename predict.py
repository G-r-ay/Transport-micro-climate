import torch
import joblib
from torch import nn
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Load the saved model weights
selected_columns = ['fkSensorSerialId_inner',
                    'temp_inner',
                    'long_inner',
                    'lat_inner',
                    'feels_like_inner',
                    'dt_hour_inner',
                    'timestamp_hour_inner',
                    'wind_deg_inner',
                    'pressure_inner',
                    'wind_gust_inner',
                    'fkLinkSerialId',
                    'sunrise_minute_inner',
                    'msla_inner',
                    'Compiled_Loaction_inner',
                    'humidity%_inner',
                    'heading_inner',
                    'wind_speed_inner',
                    'sunset_minute_inner',
                    'timestamp_day_inner',
                    'visibility_inner',
                    'clouds_inner']


class TempHumidModel(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(TempHumidModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


model_weights_path = 'Model_weights/inner_temperature_humidity.pt'
model = TempHumidModel(21, 1024)
model.load_state_dict(torch.load(model_weights_path,
                      map_location=torch.device('cpu')))
model.eval()

# Load the ordinal encoder and scaler
ordinal_encoder = joblib.load('tools/ordinal_encoder.pkl')
scaler = joblib.load('tools/scaler.pkl')

# Function to preprocess new input data


def preprocess_input(data):
    dataframe_encoded = pd.DataFrame(data, columns=selected_columns)
    object_columns = dataframe_encoded.select_dtypes(
        include=['object']).columns
    dataframe_encoded[object_columns] = ordinal_encoder.transform(
        dataframe_encoded[object_columns])
    return dataframe_encoded

# Function to make predictions


def predict_temperature_humidity(data):
    # Preprocess the input data
    preprocessed_data = preprocess_input(data)

    # Apply feature scaling
    scaled_data = scaler.transform(preprocessed_data[selected_columns])
    scaled_data = pd.DataFrame(scaled_data, columns=selected_columns)

    # Convert the input data to PyTorch tensors
    input_tensor = torch.tensor(scaled_data.values, dtype=torch.float32)

    # Make predictions
    with torch.inference_mode():
        predictions = model(input_tensor)

    # Return the predictions
    return predictions.tolist()

# Function to get user input as a dataframe or manually for each column


def get_user_input():
    input_type = input(
        "Enter '1' to input data as a dataframe, or '2' to input manually: ")

    if input_type == '1':
        # Input data as a dataframe
        file_path = input("Enter the path to the data file (CSV format): ")
        dataframe = pd.read_csv(file_path)
        return dataframe
    elif input_type == '2':
        # Manually input data for each column
        input_data = {}
        for column in selected_columns:
            value = input(f"Enter value for column '{column}': ")
            input_data[column] = [value]
        return pd.DataFrame(input_data)
    else:
        print("Invalid input. Please try again.")
        return get_user_input()


# Get user input
input_data = get_user_input()

# Make predictions using the user input
predictions = predict_temperature_humidity(input_data)
print(predictions)
