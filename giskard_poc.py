import giskard
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and train
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestClassifier().fit(X_train, y_train)

# Wrap model
giskard_model = giskard.Model(
    model=model,
    model_type="classification",
    name="iris_rf_model",
    feature_names=X.columns.tolist(),
    classification_labels=list(iris.target_names)
)

# Upload to self-hosted Giskard
giskard.upload(
    model=giskard_model,
    dataset=X_test,
    target=y_test,
    project_key="your-project-key",  # Replace with actual key from UI
    host="https://giskard-giskard-poc.apps.<your-cluster-domain>"
)
