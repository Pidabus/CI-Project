"""
Fuzzy logic module for the Student Performance prediction project.
Maps a predicted final grade (G3) to a numerical academic risk level using
a Mamdani Fuzzy Inference System (skfuzzy).
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def assess_student_risk(predicted_score):
    """
    Assess academic risk from a predicted G3 score using fuzzy logic.

    Args:
        predicted_score: Predicted final grade (expected range 0–20).

    Returns:
        risk_level: Numerical risk score (0–100) from the fuzzy system.
    """
    # --- Fuzzy variables ---
    score = ctrl.Antecedent(np.arange(0, 21, 1), "score")
    risk = ctrl.Consequent(np.arange(0, 101, 1), "risk")

    # --- Membership functions for score (G3) ---
    score["Very Low"] = fuzz.trimf(score.universe, [0, 0, 8])
    score["Low"] = fuzz.trimf(score.universe, [5, 9, 12])
    score["Medium"] = fuzz.trimf(score.universe, [10, 13, 16])
    score["High"] = fuzz.trimf(score.universe, [14, 20, 20])

    # --- Membership functions for risk ---
    risk["At-Risk"] = fuzz.trimf(risk.universe, [0, 0, 35])
    risk["Borderline"] = fuzz.trimf(risk.universe, [25, 45, 60])
    risk["Progressing"] = fuzz.trimf(risk.universe, [50, 65, 80])
    risk["On-Track"] = fuzz.trimf(risk.universe, [70, 100, 100])

    # --- Fuzzy rules ---
    rule_very_low = ctrl.Rule(score["Very Low"], risk["At-Risk"])
    rule_low = ctrl.Rule(score["Low"], risk["Borderline"])
    rule_medium = ctrl.Rule(score["Medium"], risk["Progressing"])
    rule_high = ctrl.Rule(score["High"], risk["On-Track"])

    # --- Mamdani inference system ---
    risk_ctrl = ctrl.ControlSystem(
        [rule_very_low, rule_low, rule_medium, rule_high]
    )
    risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

    # Clamp input to the score universe to avoid simulation errors
    clipped_score = np.clip(predicted_score, 0, 20)
    risk_sim.input["score"] = clipped_score
    risk_sim.compute()

    risk_level = risk_sim.output["risk"]
    return risk_level


if __name__ == "__main__":
    mock_predicted_score = 11.5
    risk_level = assess_student_risk(mock_predicted_score)

    print("Fuzzy risk assessment test:")
    print(f"  Predicted score: {mock_predicted_score}")
    print(f"  Risk level:      {risk_level:.2f}")
