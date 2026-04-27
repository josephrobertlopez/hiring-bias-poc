"""Functional validation of SkillRulesEngine on Resume Dataset."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import traceback

from src.data_processing.resume_processor import load_resume_dataset, load_bias_in_bios_dataset
from src.rules.engine import SkillRulesEngine


def run_functional_validation():
    """Run Phase 1 functional validation."""
    print("=== Phase 1: Functional Validation on Resume Dataset ===\n")

    # Load dataset
    try:
        resumes, labels, vocabulary = load_resume_dataset()
        print(f"✓ Loaded {len(resumes)} resumes with {len(set(labels))} categories")
        print(f"  Categories: {sorted(set(labels))[:10]}{'...' if len(set(labels)) > 10 else ''}\n")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        traceback.print_exc()
        return False

    # Show category distribution
    category_counts = Counter(labels)
    print("Category distribution (top 10):")
    for category, count in category_counts.most_common(10):
        print(f"  {category}: {count}")
    print()

    # Train-test split (stratified)
    try:
        train_resumes, test_resumes, train_labels, test_labels = train_test_split(
            resumes, labels, test_size=0.2, random_state=42, stratify=labels
        )
    except ValueError as e:
        # If stratification fails due to small categories, use non-stratified split
        print(f"Warning: Could not stratify (some categories too small): {e}")
        train_resumes, test_resumes, train_labels, test_labels = train_test_split(
            resumes, labels, test_size=0.2, random_state=42
        )

    print(f"Train set: {len(train_resumes)} resumes")
    print(f"Test set: {len(test_resumes)} resumes\n")

    # Convert labels to binary for each category (one-vs-rest)
    unique_categories = sorted(set(labels))
    results = {}
    errors = []

    # Test top categories first (at least 10 samples)
    test_categories = [
        cat for cat in unique_categories
        if sum(1 for l in train_labels if l == cat) >= 10
    ][:5]

    print(f"Testing {len(test_categories)} categories:\n")

    for category in test_categories:
        print(f"--- Testing category: {category} ---")

        # Create binary labels
        train_binary = [1 if label == category else 0 for label in train_labels]
        test_binary = [1 if label == category else 0 for label in test_labels]

        positive_count = sum(train_binary)
        print(f"  Training: {positive_count} positive, {len(train_binary) - positive_count} negative")

        # Skip if too few positive examples
        if positive_count < 3:
            print(f"  Skipping (only {positive_count} training examples)")
            continue

        # Train SkillRulesEngine
        try:
            engine = SkillRulesEngine(vocabulary)
            engine.fit(train_resumes, train_binary)

            # Predict on test set
            predictions = []
            scores = []

            for resume in test_resumes:
                try:
                    audit_result = engine.audit_resume(resume)
                    score = audit_result.overall_score
                    prediction = 1 if score > 0.5 else 0

                    predictions.append(prediction)
                    scores.append(score)
                except Exception as e:
                    print(f"  Warning: Error scoring resume: {e}")
                    continue

            if len(predictions) == 0:
                print(f"  Skipping (no valid predictions)")
                continue

            # Calculate metrics
            accuracy = accuracy_score(test_binary, predictions)
            test_positive = sum(test_binary)

            if sum(predictions) > 0:
                precision = precision_score(test_binary, predictions, zero_division=0)
            else:
                precision = 0.0

            if test_positive > 0:
                recall = recall_score(test_binary, predictions, zero_division=0)
            else:
                recall = 0.0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            results[category] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "test_samples": len(test_binary),
                "test_positive": test_positive,
                "predictions_positive": sum(predictions),
                "avg_score": float(np.mean(scores)),
                "std_score": float(np.std(scores))
            }

            print(f"  Accuracy:  {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall:    {recall:.3f}")
            print(f"  F1 Score:  {f1:.3f}")
            print(f"  Avg score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
            print()

        except Exception as e:
            error_msg = f"Error training {category}: {str(e)}"
            print(f"  ✗ {error_msg}")
            errors.append(error_msg)
            traceback.print_exc()
            continue

    # Summary
    print("=== FUNCTIONAL VALIDATION SUMMARY ===\n")

    if results:
        accuracies = [r["accuracy"] for r in results.values()]
        f1_scores = [r["f1"] for r in results.values()]

        avg_accuracy = np.mean(accuracies)
        avg_f1 = np.mean(f1_scores)

        print(f"Categories tested: {len(results)}")
        print(f"Average accuracy:  {avg_accuracy:.3f}")
        print(f"Average F1 score:  {avg_f1:.3f}")
        print(f"Pass threshold:    0.6 (random baseline ~{1.0 / (len(set(labels))):.3f})")

        if avg_accuracy >= 0.6:
            print(f"Status: ✓ PASS\n")
            success = True
        else:
            print(f"Status: ✗ FAIL\n")
            success = False

        # Save results
        output_path = Path("results/functional_validation.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "summary": {
                    "categories_tested": len(results),
                    "avg_accuracy": avg_accuracy,
                    "avg_f1": avg_f1,
                    "pass_threshold": 0.6,
                    "passed": success
                },
                "results": results,
                "errors": errors
            }, f, indent=2)

        print(f"Results saved to: {output_path}")
        return success

    else:
        print("✗ No categories could be tested")
        if errors:
            print("\nErrors encountered:")
            for error in errors:
                print(f"  - {error}")
        return False


def run_bias_in_bios_validation():
    """Run functional validation on Bias-in-Bios dataset if available."""
    print("\n=== Bonus: Bias-in-Bios Dataset Validation ===\n")

    result = load_bias_in_bios_dataset()
    if result is None:
        print("Bias-in-Bios dataset not available (skipping)")
        return True

    resumes, labels, vocabulary = result
    print(f"✓ Loaded {len(resumes)} examples with {len(set(labels))} occupations\n")

    # Sample top occupations
    occupation_counts = Counter(labels)
    test_occupations = [occ for occ, count in occupation_counts.most_common(3) if count >= 5]

    if not test_occupations:
        print("Insufficient samples for testing")
        return True

    print(f"Testing top occupations: {test_occupations}\n")

    for occupation in test_occupations:
        print(f"--- {occupation} ---")
        try:
            train_resumes, test_resumes, train_labels, test_labels = train_test_split(
                resumes, labels, test_size=0.2, random_state=42
            )

            train_binary = [1 if label == occupation else 0 for label in train_labels]
            test_binary = [1 if label == occupation else 0 for label in test_labels]

            if sum(train_binary) < 3:
                print(f"Skipping (only {sum(train_binary)} training examples)\n")
                continue

            engine = SkillRulesEngine(vocabulary)
            engine.fit(train_resumes, train_binary)

            predictions = []
            for resume in test_resumes:
                try:
                    audit_result = engine.audit_resume(resume)
                    predictions.append(1 if audit_result.overall_score > 0.5 else 0)
                except Exception:
                    continue

            if predictions:
                accuracy = accuracy_score(test_binary, predictions)
                print(f"Accuracy: {accuracy:.3f}\n")

        except Exception as e:
            print(f"Error: {e}\n")

    return True


if __name__ == "__main__":
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        success = run_functional_validation()
        run_bias_in_bios_validation()

        if success:
            print("\n✓ Phase 1 Validation Complete - Engine is functional")
            sys.exit(0)
        else:
            print("\n✗ Phase 1 Validation Complete - Engine needs improvement")
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
