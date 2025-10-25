from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# split dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# train random forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train,y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print(f"Model Accuracy: {accuracy}")
print("Classification Report:")
print(report)

joblib.dump(clf, "iris_model.pkl")