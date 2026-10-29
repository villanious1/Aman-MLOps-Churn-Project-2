def test_training_saves_artifacts(tmp_path, monkeypatch):
    import src.churn_ml.train as training
    monkeypatch.setattr(training, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(training, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(training, "METRICS_PATH", tmp_path / "metrics.json")
    report = training.train_and_evaluate(n_samples=160, random_state=12)
    assert report["selected_model"] in report["models"]
    assert training.MODEL_PATH.exists()
    assert training.METRICS_PATH.exists()
    for scores in report["models"].values():
        assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= scores.keys()
