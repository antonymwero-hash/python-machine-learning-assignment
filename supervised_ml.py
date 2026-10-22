import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
y = np.array([35,40,45,50,55,60,65,70])

model = LinearRegression()
model.fit(X, y)
pred = model.predict([[9]])
print(f"Predicted score for 9 hours: {pred[0]:.2f}")

plt.scatter(X, y, label='Actual')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.title('Supervised Learning: Linear Regression')
plt.savefig('supervised_plot.png')
