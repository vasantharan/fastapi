import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
np.random.seed(0)
x_train = np.random.rand(100, 1)
y_train = 3.5 * x_train + np.random.randn(100, 1) * 0.2
model = LinearRegression()
model.fit(x_train, y_train)
joblib.dump(model, 'linear_regression_model.pkl')
