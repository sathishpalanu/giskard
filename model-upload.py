giskard.upload(
    model=giskard_model,
    dataset=X_test,
    target=y_test,
    project_key="your-project-key",
    host="https://giskard-giskard-poc.apps.<your-cluster-domain>"
)
