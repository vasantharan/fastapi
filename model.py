import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
x = iris.data
y = iris.target

rf_model = RandomForestClassifier()
rf_model.fit(x,y)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x,y)

joblib.dump(rf_model, './api/Random_Forest.joblib')
joblib.dump(knn_model, './api/KNN.joblib')