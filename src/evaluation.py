def evaluate_predictions(
    X_train, X_test, y_train, y_test, estimator, best_params, default_params, metric
):
    """
    Evaluate predictions
    """
    model = estimator(**best_params, **default_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = metric(y_true=y_test, y_pred=y_pred)
    return score
