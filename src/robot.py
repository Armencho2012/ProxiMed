import pandas as pd
from model import DiabetesKNN

data = pd.read_csv('database.csv')
model = DiabetesKNN(data)

def test_robot(values):
    attr_map = dict(zip(model.attributes, values))
    return model.predict(attr_map, k=5)

# Automated test case
sample_input = [1.0, 1.0, 30.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 6.0, 6.0, 8.0]
result = test_robot(sample_input)
print(f"Robot Prediction: {int(result)}")