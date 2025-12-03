import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.DataFrame({
    'Engine_CC': [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500],
    'Car_Weight': [700, 800, 900, 1000, 1200, 1300, 1500, 1600],
    'Mileage': [22, 20, 19, 17, 15, 14, 13, 11] 
})
X = data[['Engine_CC', 'Car_Weight']]
y = data['Mileage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print("\n=== Car Mileage Predictor ===")
engine = float(input("Enter engine size (in CC): "))
weight = float(input("Enter car weight (in KG): "))
pred = model.predict([[engine, weight]])[0]
print(f"\nEstimated Mileage: {pred:.2f} km/l")
