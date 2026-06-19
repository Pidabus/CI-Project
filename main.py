"""
Main integration script for the Student Performance risk prediction project.

Data flow:
  1. data_preprocessing  → load, encode, scale, and split the dataset
  2. data_mlp            → train MLP via GridSearchCV and predict on test set
  3. data_fuzzy          → map predicted grade to a fuzzy risk level
"""

from data_preprocessing import load_and_preprocess
from data_mlp import train_and_evaluate_mlp
from data_fuzzy import assess_student_risk


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Step 1: Data preprocessing
    # Load student-mat.csv, encode categoricals, scale features, and split
    # into 85% training / 15% testing sets.
    # -------------------------------------------------------------------------
    print("=" * 50)
    print("STUDENT RISK PREDICTION SYSTEM")
    print("=" * 50)
    print("\n[1/3] Loading and preprocessing data...")

    X_train, X_test, y_train, y_test = load_and_preprocess()

    print(f"      Training samples: {len(X_train)}")
    print(f"      Test samples:     {len(X_test)}")

    # -------------------------------------------------------------------------
    # Step 2: MLP training and evaluation
    # GridSearchCV finds the best hyperparameters, then predicts G3 on X_test.
    # -------------------------------------------------------------------------
    print("\n[2/3] Training MLP neural network (grid search may take a few minutes)...")

    best_model, y_pred = train_and_evaluate_mlp(X_train, X_test, y_train, y_test)

    # -------------------------------------------------------------------------
    # Step 3: Fuzzy risk assessment for the first test-set student
    # Index 0 is the first student in the held-out test split.
    # -------------------------------------------------------------------------
    print("\n[3/3] Assessing risk for the first test-set student...")

    student_index = 50
    actual_grade = y_test.iloc[student_index]
    predicted_grade = y_pred[student_index]

    risk_level = assess_student_risk(predicted_grade)

    # Convert fuzzy score into a readable category
    if risk_level < 35:
        risk_category = "At-Risk"
    elif risk_level < 60:
        risk_category = "Borderline"
    elif risk_level < 80:
        risk_category = "Progressing"
    else:
        risk_category = "On-Track"

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("FINAL STUDENT RISK REPORT")
    print("=" * 50)
    print(f"  Student index (test set): {student_index}")
    print(f"  Actual G3 grade:          {actual_grade:.2f}")
    print(f"  Predicted G3 grade:       {predicted_grade:.2f}")
    print(f"  Fuzzy risk level:         {risk_level:.2f}")
    print(f"  Risk category:            {risk_category}")
    print("=" * 50)
