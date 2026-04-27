"""Phase 2: Bias Detection Validation on Bias-in-Bios Dataset."""
import sys
sys.path.insert(0, '/home/joey/Documents/GitHub/hiring-bias-poc')

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.data_processing.resume_processor import ResumeProcessor
from src.rules.data import Resume
from src.rules.engine import SkillRulesEngine


class BiasInBiosProcessor(ResumeProcessor):
    """Process Bias-in-Bios dataset into Resume format with demographic labels."""

    def __init__(self):
        super().__init__()

        # Profession ID to name mapping (from Bias-in-Bios paper)
        self.profession_names = {
            0: "Accountant", 1: "Architect", 2: "Attorney", 3: "Chiropractor", 4: "Comedian",
            5: "Composer", 6: "Dentist", 7: "Dietitian", 8: "DJ", 9: "Filmmaker",
            10: "Interior_designer", 11: "Journalist", 12: "Model", 13: "Nurse", 14: "Painter",
            15: "Paralegal", 16: "Pastor", 17: "Personal_trainer", 18: "Photographer", 19: "Physician",
            20: "Poet", 21: "Professor", 22: "Psychologist", 23: "Rapper", 24: "Software_engineer",
            25: "Surgeon", 26: "Teacher", 27: "Yoga_teacher"
        }

        # Gender mapping: 0 = male, 1 = female
        self.gender_names = {0: "male", 1: "female"}

    def extract_skills_from_bio(self, bio_text: str) -> List[str]:
        """Extract skills from biographical text using parent vocabulary."""
        if not isinstance(bio_text, str) or len(bio_text.strip()) == 0:
            return []

        found_skills = []
        text_lower = bio_text.lower()

        # Use parent class vocabulary with word boundary regex
        for token in self.vocabulary.tokens:
            pattern = r'\b' + re.escape(token.lower().replace(' ', r'\s+')) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(token)

        return found_skills

    def extract_experience_from_bio(self, bio_text: str) -> float:
        """Extract years of experience from bio text using regex patterns."""
        if not isinstance(bio_text, str):
            return 0.0

        # Look for patterns like "5 years", "3+ years", "2-4 years"
        patterns = [
            r'(\d+)\+?\s+years?\s+(?:of\s+)?experience',
            r'experience\s+(?:of\s+)?(\d+)\+?\s+years?',
            r'(\d+)\+?\s+years?\s+in',
            r'(\d+)-\d+\s+years?',
            r'(\d+)\s+years?',
        ]

        years = []
        for pattern in patterns:
            matches = re.findall(pattern, bio_text.lower())
            years.extend([int(m) for m in matches])

        return float(max(years)) if years else 0.0

    def _profession_to_domain(self, profession: str) -> str:
        """Map profession to domain background."""
        domain_mapping = {
            # Technology
            "Software_engineer": "technology",
            "DJ": "entertainment",
            "Filmmaker": "entertainment",

            # Healthcare
            "Physician": "healthcare",
            "Surgeon": "healthcare",
            "Nurse": "healthcare",
            "Dentist": "healthcare",
            "Chiropractor": "healthcare",
            "Dietitian": "healthcare",
            "Psychologist": "healthcare",

            # Education
            "Professor": "education",
            "Teacher": "education",

            # Legal/Finance
            "Attorney": "legal",
            "Paralegal": "legal",
            "Accountant": "finance",

            # Creative
            "Architect": "design",
            "Interior_designer": "design",
            "Painter": "arts",
            "Composer": "arts",
            "Poet": "arts",
            "Photographer": "arts",
            "Model": "entertainment",
            "Comedian": "entertainment",
            "Rapper": "entertainment",

            # Services
            "Personal_trainer": "fitness",
            "Yoga_teacher": "fitness",
            "Pastor": "religious",
            "Journalist": "media"
        }

        return domain_mapping.get(profession, "other")

    def process_bias_in_bios(self, data_path: str, sample_size: int = 2000) -> Tuple[List[Resume], List[int], List[str]]:
        """Process Bias-in-Bios dataset into Resume objects with demographic labels.

        Args:
            data_path: Path to bias_in_bios CSV file
            sample_size: Number of examples to process

        Returns:
            Tuple of (resumes, gender_labels, profession_labels)
        """
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}\nRun: python scripts/download_datasets.py")

        print(f"Loading Bias-in-Bios dataset from {data_path}")
        df = pd.read_csv(data_path)

        # Sample for faster processing
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} examples for processing")

        # Process each bio
        resumes = []
        gender_labels = []
        profession_labels = []

        processed = 0
        skipped = 0

        for idx, row in df.iterrows():
            try:
                bio_text = str(row.get('bio', row.get('hard_text', '')))
                profession_id = int(row['profession']) if 'profession' in row else 0
                gender_id = int(row['gender']) if 'gender' in row else 0

                if len(bio_text.strip()) < 10:
                    skipped += 1
                    continue

                # Extract features from bio
                skills = self.extract_skills_from_bio(bio_text)
                experience = self.extract_experience_from_bio(bio_text)
                education = self.extract_education(bio_text)

                # Map profession to domain
                profession_name = self.profession_names.get(profession_id, "Unknown")
                domain = self._profession_to_domain(profession_name)

                # Create Resume object
                resume = Resume(
                    skill_tokens=skills,
                    years_experience=experience,
                    education_level=education,
                    domain_background=[domain] if domain else [],
                    demographics={
                        "gender": gender_id,  # 0=male, 1=female
                        "profession_id": profession_id,
                        "profession_name": profession_name
                    }
                )

                resumes.append(resume)
                gender_labels.append(gender_id)
                profession_labels.append(profession_name)

                processed += 1
                if processed % 500 == 0:
                    print(f"Processed {processed} bios...")

            except Exception as e:
                # print(f"Warning: Error processing row {idx}: {e}")
                skipped += 1
                continue

        print(f"Successfully processed {len(resumes)} bios ({skipped} skipped)")
        return resumes, gender_labels, profession_labels


def create_synthetic_hiring_outcomes(resumes: List[Resume], profession_name: str) -> List[bool]:
    """Create synthetic hiring outcomes with realistic gender bias.

    Base hiring probability is derived from skills and experience.
    Gender bias is applied: women in tech get lower probability, men in nursing get lower probability.

    Args:
        resumes: List of Resume objects for a single profession
        profession_name: Name of profession (e.g., "Software_engineer")

    Returns:
        List of bool: True = hired, False = rejected
    """
    hiring_outcomes = []

    for resume in resumes:
        # Base probability: more skills and experience → higher hiring probability
        base_prob = min(0.8, (len(resume.skill_tokens) * 0.1 + resume.years_experience * 0.05))

        # Apply gender bias for specific professions
        gender = resume.demographics.get("gender", 0)  # 0=male, 1=female
        bias_factor = 1.0

        if profession_name == "Software_engineer" and gender == 1:  # Female in tech
            bias_factor = 0.7  # 30% bias against females
        elif profession_name == "Nurse" and gender == 0:  # Male in nursing
            bias_factor = 0.8  # 20% bias against males
        elif profession_name == "Teacher" and gender == 0:  # Male in teaching
            bias_factor = 0.85  # 15% bias against males

        final_prob = min(1.0, base_prob * bias_factor)
        hired = np.random.random() < final_prob
        hiring_outcomes.append(hired)

    return hiring_outcomes


def run_bias_detection_validation():
    """Run Phase 2: Bias detection validation on Bias-in-Bios dataset."""
    print("="*60)
    print("Phase 2: Bias Detection Validation on Bias-in-Bios")
    print("="*60)

    # Setup
    processor = BiasInBiosProcessor()
    data_path = "/home/joey/Documents/GitHub/hiring-bias-poc/data/raw/bias_in_bios/train.csv"

    if not Path(data_path).exists():
        print(f"\nDataset not found at {data_path}")
        print("Please download first using: python scripts/download_datasets.py")
        return

    # Process dataset
    try:
        resumes, gender_labels, profession_labels = processor.process_bias_in_bios(
            data_path, sample_size=2000
        )

        print(f"\nDataset Summary:")
        print(f"  Total resumes: {len(resumes)}")
        print(f"  Gender distribution: {Counter(gender_labels)}")
        print(f"  Top professions: {Counter(profession_labels).most_common(5)}")

    except Exception as e:
        print(f"\nFailed to process dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Focus on gender bias detection for specific professions
    target_professions = ["Software_engineer", "Nurse", "Teacher", "Physician", "Attorney"]

    results = {}

    for profession in target_professions:
        print(f"\n--- Analyzing {profession} for gender bias ---")

        # Filter to this profession only
        prof_indices = [i for i, p in enumerate(profession_labels) if p == profession]
        if len(prof_indices) < 20:
            print(f"Skipping {profession} - only {len(prof_indices)} samples")
            continue

        prof_resumes = [resumes[i] for i in prof_indices]
        prof_genders = [gender_labels[i] for i in prof_indices]

        print(f"Profession samples: {len(prof_resumes)}")
        print(f"Gender breakdown: {Counter(prof_genders)}")

        # Create synthetic hiring outcomes with bias
        hiring_outcomes = create_synthetic_hiring_outcomes(prof_resumes, profession)
        print(f"Synthetic hiring outcomes: {Counter(hiring_outcomes)}")

        # Split train-test
        train_resumes, test_resumes, train_labels, test_labels = train_test_split(
            prof_resumes, hiring_outcomes, test_size=0.3, random_state=42
        )

        # Train SkillRulesEngine
        try:
            engine = SkillRulesEngine(processor.vocabulary)
            engine.fit(train_resumes, train_labels)

            # Test and collect predictions
            predictions = []
            bias_flags_found = []

            for resume in test_resumes:
                audit_result = engine.audit_resume(resume)
                predictions.append(1 if audit_result.overall_score > 0.5 else 0)

                # Check if bias was detected
                if audit_result.bias_flags:
                    bias_flags_found.append(True)
                else:
                    bias_flags_found.append(False)

            # Calculate metrics by gender
            test_genders = [resume.demographics["gender"] for resume in test_resumes]

            male_indices = [i for i, g in enumerate(test_genders) if g == 0]
            female_indices = [i for i, g in enumerate(test_genders) if g == 1]

            male_hiring_rate = (
                np.mean([predictions[i] for i in male_indices]) if male_indices else 0
            )
            female_hiring_rate = (
                np.mean([predictions[i] for i in female_indices]) if female_indices else 0
            )

            # Disparity index (4/5ths rule)
            if male_hiring_rate > 0 and female_hiring_rate > 0:
                disparity_index = min(male_hiring_rate, female_hiring_rate) / max(male_hiring_rate, female_hiring_rate)
            elif male_hiring_rate == 0 and female_hiring_rate == 0:
                disparity_index = 1.0
            else:
                disparity_index = 0.0

            # Overall accuracy
            accuracy = accuracy_score(test_labels, predictions)

            results[profession] = {
                "accuracy": float(accuracy),
                "male_hiring_rate": float(male_hiring_rate),
                "female_hiring_rate": float(female_hiring_rate),
                "disparity_index": float(disparity_index),
                "bias_detected": disparity_index < 0.8,
                "bias_flags_count": int(sum(bias_flags_found)),
                "test_samples": len(test_resumes),
                "male_count": len(male_indices),
                "female_count": len(female_indices)
            }

            print(f"Accuracy: {accuracy:.3f}")
            print(f"Male hiring rate: {male_hiring_rate:.3f} ({len(male_indices)} samples)")
            print(f"Female hiring rate: {female_hiring_rate:.3f} ({len(female_indices)} samples)")
            print(f"Disparity Index: {disparity_index:.3f}")
            print(f"Bias detected: {'YES' if disparity_index < 0.8 else 'NO'}")
            print(f"Bias flags fired: {sum(bias_flags_found)}/{len(bias_flags_found)}")

        except Exception as e:
            print(f"Error analyzing {profession}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary report
    print(f"\n{'='*60}")
    print(f"BIAS DETECTION VALIDATION SUMMARY")
    print(f"{'='*60}")

    if results:
        avg_accuracy = np.mean([r["accuracy"] for r in results.values()])
        professions_with_bias = [p for p, r in results.items() if r["bias_detected"]]

        print(f"Professions tested: {len(results)}")
        print(f"Average accuracy: {avg_accuracy:.3f}")
        print(f"Professions with detected bias: {professions_with_bias}")

        # Detailed results
        print(f"\nDetailed Results:")
        print(f"{'Profession':<20} {'Accuracy':<12} {'M-Rate':<12} {'F-Rate':<12} {'D-Index':<12} {'Bias?':<8}")
        print("-" * 80)
        for profession, metrics in results.items():
            print(f"{profession:<20} {metrics['accuracy']:<12.3f} {metrics['male_hiring_rate']:<12.3f} {metrics['female_hiring_rate']:<12.3f} {metrics['disparity_index']:<12.3f} {'YES' if metrics['bias_detected'] else 'NO':<8}")

        # Save results
        Path("results").mkdir(exist_ok=True)
        results_path = Path("results/bias_detection_validation.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_path}")
    else:
        print("No professions could be analyzed")


if __name__ == "__main__":
    run_bias_detection_validation()
