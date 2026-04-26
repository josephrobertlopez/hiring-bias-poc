Feature: Ensemble - XGBoost + Ensemble Methods for Accuracy Boost with Bias Constraints

  Background:
    Given an ensemble module is initialized
    And random seed is fixed to 42

  @contract @E001
  Scenario: XGBoost baseline model training
    Given Adult dataset with 5K training records
    And target variable (hired/not hired)
    And features (age, education, experience, skills)
    When I train XGBoost baseline
    Then training succeeds without error
    And model is fitted (supports predict/predict_proba)
    And AUC on held-out test set > 0.70
    And hyperparameters are logged and reproducible

  @contract @E002
  Scenario: Voting ensemble combining models
    Given trained XGBoost, LogisticRegression, RandomForest models
    And 1000 test records
    When I create voting ensemble (hard voting by default)
    Then predictions are integer labels {0, 1}
    And voting respects majority class per record
    And AUC >= max(component model AUCs) - 0.02
    And ensemble diversity improves some test instances

  @contract @E003
  Scenario: Soft voting with confidence aggregation
    Given 3 models that support predict_proba
    And test data with 1000 records
    When I create soft voting ensemble (average probabilities)
    Then predictions are probabilities in [0, 1]
    And averaging respects equal weights (configurable)
    And AUC >= median(component AUCs)
    And soft ensemble smoother than hard voting

  @contract @E004
  Scenario: Stacking ensemble with meta-learner
    Given XGBoost, RandomForest, LogisticRegression base models
    And 1000 training records
    When I train stacking ensemble with meta-learner
    Then meta-learner is fitted on base model outputs
    And stacking AUC >= best base model AUC
    And stacking generalizes without overfitting
    And meta-learner combines base predictions effectively

  @contract @E005
  Scenario: Per-task hyperparameter tuning
    Given Adult hiring task with 5K records
    And Bias-in-Bios task with 10K records
    When I tune XGBoost separately per task
    Then Adult model uses Adult-optimized parameters
    And Bias-in-Bios model uses Bios-optimized parameters
    And tuned AUC > baseline AUC by >= 0.03
    And parameter importance varies per task

  @contract @E006
  Scenario: Fairness constraints during ensemble training
    Given ensemble training data with protected attribute
    And fairness_v2 demographic parity constraint (DI >= 0.8)
    When I train ensemble with constraint
    Then ensemble respects fairness constraint
    And fairness metric >= 0.8 on validation set
    And accuracy loss vs unconstrained is < 0.05
    And bias constraint is verifiable

  @contract @E007
  Scenario: Feature importance ranking
    Given trained ensemble model
    When I compute feature importance
    Then importance is numeric >= 0
    And features ranked by importance
    And top features align with domain knowledge
    And importance visualization is interpretable
    And importance includes base model contribution

  @contract @E008
  Scenario: SHAP explainability integration
    Given trained ensemble model
    And 100 test instances
    When I compute SHAP values
    Then SHAP values match prediction magnitude
    And explanation supports model output
    And feature contribution is per-instance
    And SHAP sums approximately to predicted value - base_value
    And visualization is possible (force plot, summary plot)

  @contract @E009
  Scenario: Cross-validation evaluation
    Given Adult dataset with 5K records
    And ensemble configuration
    When I run 5-fold cross-validation
    Then AUC reported per fold
    And mean AUC and std reported
    And results are reproducible with seed
    And ensemble trains 5 times (once per fold)

  @contract @E010
  Scenario: Ensemble reproducibility with seed control
    Given ensemble initialized with seed=42
    When I train ensemble model
    And re-initialize with same seed
    And retrain ensemble
    Then second model produces identical predictions
    And hyperparameter search order is deterministic
    And reproducibility holds across training runs

