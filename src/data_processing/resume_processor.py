"""Process Resume Dataset into SkillRulesEngine format."""
import pandas as pd
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from ..rules.data import Resume, SkillVocabulary


class ResumeProcessor:
    """Convert raw resume data to Resume dataclass format."""

    def __init__(self):
        # Technical skills vocabulary (expandable)
        self.tech_skills = [
            # Programming languages
            "python", "java", "javascript", "c++", "c#", "r", "sql", "scala", "go", "rust",
            "typescript", "swift", "kotlin", "php", "ruby", "perl", "matlab", "vba",
            "lua", "groovy", "dart", "elixir", "haskell", "clojure", "f#",

            # Web Frameworks and libraries
            "react", "angular", "vue", "node.js", "django", "flask", "spring", "express",
            "fastapi", "laravel", "rails", "asp.net", "tomcat", "nginx", "apache",

            # ML and Data libraries
            "tensorflow", "pytorch", "sklearn", "pandas", "numpy", "matplotlib", "seaborn",
            "keras", "xgboost", "lightgbm", "scipy", "statsmodels", "plotly",

            # Databases
            "mysql", "postgresql", "mongodb", "redis", "cassandra", "oracle", "sqlite",
            "dynamodb", "firestore", "elasticsearch", "memcached", "neo4j", "hbase",

            # Cloud & DevOps
            "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "terraform",
            "ansible", "puppet", "chef", "circleci", "gitlab", "github", "bitbucket",
            "cloudformation", "sam", "serverless",

            # Data & Analytics
            "hadoop", "spark", "kafka", "airflow", "tableau", "powerbi", "excel",
            "looker", "dbt", "dataflow", "databricks", "snowflake", "bigquery",

            # Web Technologies
            "html", "css", "rest api", "graphql", "microservices", "soap", "xml", "json",
            "websocket", "grpc", "protobuf", "jwt", "oauth",

            # Mobile
            "ios", "android", "react native", "flutter", "xamarin",

            # Testing & QA
            "junit", "pytest", "testng", "selenium", "cucumber", "jmeter", "postman",
            "mockito", "jasmine", "mocha", "rspec",

            # Version Control & Collaboration
            "git", "svn", "perforce", "jira", "confluence", "slack", "trello",

            # Other Tools
            "linux", "unix", "windows", "bash", "powershell", "vim", "emacs",
            "intellij", "vscode", "eclipse", "visual studio"
        ]

        # Soft skills
        self.soft_skills = [
            "leadership", "communication", "teamwork", "problem solving", "project management",
            "analytical thinking", "creativity", "adaptability", "time management",
            "critical thinking", "collaboration", "negotiation", "presentation",
            "mentoring", "conflict resolution", "decision making", "strategic thinking",
            "attention to detail", "work ethic", "reliability", "accountability"
        ]

        # Combined vocabulary
        all_skills = self.tech_skills + self.soft_skills
        self.vocabulary = SkillVocabulary(
            tokens=all_skills,
            categories={
                "programming": ["python", "java", "javascript", "c++", "c#", "r", "sql",
                               "scala", "go", "rust", "typescript", "swift", "kotlin", "php"],
                "web": ["html", "css", "javascript", "react", "angular", "vue", "node.js",
                       "express", "django", "flask", "spring", "rest api", "graphql"],
                "data": ["python", "r", "sql", "tensorflow", "pytorch", "sklearn", "pandas",
                        "numpy", "spark", "hadoop", "tableau", "powerbi"],
                "cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform",
                         "ansible", "cloudformation"],
                "devops": ["docker", "kubernetes", "jenkins", "git", "ansible", "terraform",
                          "circleci", "gitlab", "github"],
                "soft": self.soft_skills
            }
        )

    # Common English words to skip during fuzzy matching
    _STOP_WORDS = {
        'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'been',
        'will', 'are', 'was', 'has', 'not', 'but', 'all', 'can', 'its', 'our',
        'your', 'their', 'they', 'than', 'over', 'into', 'also', 'work', 'used',
        'use', 'using', 'team', 'years', 'year', 'able', 'good', 'high', 'well',
        'new', 'one', 'two', 'three', 'four', 'five', 'six', 'strong', 'about',
    }

    def extract_skills(self, resume_text: str) -> List[str]:
        """Extract skills via exact regex match. Fast — used for training data."""
        if not isinstance(resume_text, str):
            return []
        text_lower = resume_text.lower()
        found = []
        for skill in self.vocabulary.tokens:
            pattern = r'\b' + re.escape(skill.lower().replace(' ', r'\s+')) + r'\b'
            if re.search(pattern, text_lower):
                found.append(skill)
        return found

    def extract_skills_with_scores(self, resume_text: str, fuzzy_threshold: int = 82) -> Dict[str, dict]:
        """Like extract_skills but returns {skill: {'method': 'exact'|'fuzzy', 'score': int}}."""
        if not isinstance(resume_text, str):
            return {}

        text_lower = resume_text.lower()
        result: Dict[str, dict] = {}

        for skill in self.vocabulary.tokens:
            pattern = r'\b' + re.escape(skill.lower().replace(' ', r'\s+')) + r'\b'
            if re.search(pattern, text_lower):
                result[skill] = {'method': 'exact', 'score': 100}

        try:
            from rapidfuzz import process as fz_process, fuzz as fz
            found_lower = {s.lower() for s in result}
            remaining = [s for s in self.vocabulary.tokens if s.lower() not in found_lower]
            remaining_lower = [s.lower() for s in remaining]

            words = re.findall(r'\b[\w][\w.+#]*\b', text_lower)
            candidates = set(w for w in words if len(w) >= 2 and w not in self._STOP_WORDS)
            for i in range(len(words) - 1):
                candidates.add(f"{words[i]} {words[i+1]}")

            candidates = list(candidates)
            if candidates and remaining_lower:
                scores = fz_process.cdist(
                    candidates, remaining_lower,
                    scorer=fz.ratio, score_cutoff=fuzzy_threshold,
                )
                import numpy as np
                best_per_skill = scores.max(axis=0) if len(scores) else np.zeros(len(remaining_lower))
                best_candidate_idx = scores.argmax(axis=0) if len(scores) else []
                for i, best in enumerate(best_per_skill):
                    if best >= fuzzy_threshold:
                        matched_by = candidates[best_candidate_idx[i]] if len(best_candidate_idx) else '?'
                        result[remaining[i]] = {'method': 'fuzzy', 'score': int(best),
                                                'matched_by': matched_by}
        except Exception:
            pass

        return result

    def extract_experience(self, resume_text: str) -> float:
        """Extract years of experience from resume text."""
        if not isinstance(resume_text, str):
            return 0.0

        # Look for patterns like "5 years", "3+ years", "2-4 years"
        patterns = [
            r'(\d+)\+?\s+years?\s+(?:of\s+)?experience',
            r'experience\s+(?:of\s+)?(\d+)\+?\s+years?',
            r'(\d+)\+?\s+years?\s+in',
            r'(\d+)-\d+\s+years?',
        ]

        years = []
        for pattern in patterns:
            matches = re.findall(pattern, resume_text.lower())
            years.extend([int(m) for m in matches])

        return float(max(years)) if years else 0.0

    def extract_education(self, resume_text: str) -> str:
        """Extract education level from resume text."""
        if not isinstance(resume_text, str):
            return "bachelor"

        text_lower = resume_text.lower()

        # Check for education levels in order of specificity
        if any(term in text_lower for term in ['phd', 'ph.d', 'doctorate', 'doctoral']):
            return "phd"
        elif any(term in text_lower for term in ['master', 'm.s', 'm.a', 'mba', 'ms', 'ma']):
            return "master"
        elif any(term in text_lower for term in ['bachelor', 'b.s', 'b.a', 'bs', 'ba', 'undergraduate']):
            return "bachelor"
        elif any(term in text_lower for term in ['bootcamp', 'coding bootcamp', 'certificate']):
            return "bootcamp"
        else:
            return "bachelor"  # default assumption

    def extract_domain(self, resume_text: str) -> List[str]:
        """Extract domain background from resume text."""
        if not isinstance(resume_text, str):
            return ["technology"]

        text_lower = resume_text.lower()
        domains = []

        domain_keywords = {
            "finance": ["finance", "financial", "banking", "investment", "trading", "fintech",
                       "forex", "derivatives", "portfolio"],
            "healthcare": ["healthcare", "medical", "hospital", "pharmaceutical", "biotech",
                          "nursing", "clinical", "patient"],
            "technology": ["software", "tech", "it", "computer", "development", "engineering",
                          "startup", "saas"],
            "retail": ["retail", "ecommerce", "e-commerce", "sales", "marketing", "consumer",
                      "store", "merchandise"],
            "education": ["education", "academic", "university", "school", "teaching", "student",
                         "professor", "instructor"],
            "consulting": ["consulting", "advisory", "strategy", "management consulting",
                          "business consultant"],
            "manufacturing": ["manufacturing", "production", "supply chain", "logistics",
                             "warehouse", "quality assurance"],
            "government": ["government", "federal", "state", "public sector", "defense",
                          "civil service"],
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                domains.append(domain)

        return domains if domains else ["technology"]

    def process_dataset(self, csv_path: str) -> Tuple[List[Resume], List[str]]:
        """Process the Resume Dataset CSV into Resume objects."""
        df = pd.read_csv(csv_path)

        # Expected columns: 'Resume_str' and 'Category' (or similar)
        text_col = None
        category_col = None

        for col in df.columns:
            if 'resume' in col.lower() or 'text' in col.lower():
                text_col = col
            elif 'category' in col.lower() or 'label' in col.lower() or 'job' in col.lower():
                category_col = col

        if not text_col or not category_col:
            raise ValueError(
                f"Cannot find resume text and category columns in {df.columns.tolist()}"
            )

        # Process each resume
        resumes = []
        labels = []
        skipped = 0

        for idx, row in df.iterrows():
            try:
                resume_text = str(row[text_col])
                category = str(row[category_col])

                # Skip empty resumes
                if not resume_text or len(resume_text.strip()) < 10:
                    skipped += 1
                    continue

                # Extract features
                skills = self.extract_skills(resume_text)
                experience = self.extract_experience(resume_text)
                education = self.extract_education(resume_text)
                domain = self.extract_domain(resume_text)

                # Create Resume object (no demographics in this dataset)
                resume = Resume(
                    skill_tokens=skills,
                    years_experience=experience,
                    education_level=education,
                    domain_background=domain,
                    demographics={}  # Empty - no demographic data available
                )

                resumes.append(resume)
                labels.append(category)

            except Exception as e:
                print(f"Warning: Error processing row {idx}: {e}")
                skipped += 1
                continue

        print(f"Processed {len(resumes)} resumes from {len(df)} rows ({skipped} skipped)")
        return resumes, labels


def load_resume_dataset(data_dir: str = "data/raw") -> Tuple[List[Resume], List[str], SkillVocabulary]:
    """Load and process Resume Dataset from multiple sources."""
    processor = ResumeProcessor()

    # Try multiple possible file locations
    possible_paths = [
        f"{data_dir}/resume_dataset/Resume.csv",
        f"{data_dir}/resume_dataset/resume_dataset.csv",
        f"{data_dir}/resume_dataset/resumes.csv",
        f"{data_dir}/resume_dataset.csv",
    ]

    csv_path = None
    for path in possible_paths:
        if Path(path).exists():
            csv_path = path
            break

    if not csv_path:
        raise FileNotFoundError(
            f"Resume dataset CSV not found. Tried: {possible_paths}\n"
            "Run: python scripts/download_datasets.py"
        )

    print(f"Loading dataset from: {csv_path}")
    resumes, labels = processor.process_dataset(csv_path)
    return resumes, labels, processor.vocabulary


def load_bias_in_bios_dataset(data_dir: str = "data/raw") -> Optional[Tuple[List[Resume], List[str], SkillVocabulary]]:
    """Load Bias-in-Bios dataset if available."""
    try:
        from datasets import load_from_disk
        processor = ResumeProcessor()
        dataset_path = Path(data_dir) / "bias_in_bios"

        if not dataset_path.exists():
            print(f"Bias-in-Bios dataset not found at {dataset_path}")
            return None

        print(f"Loading Bias-in-Bios dataset from: {dataset_path}")
        dataset = load_from_disk(str(dataset_path))

        # Convert to Resume objects
        resumes = []
        labels = []

        for example in dataset["train"]:
            try:
                # Extract bio text
                bio = example.get("bio", "")
                if not bio or len(bio.strip()) < 10:
                    continue

                # Get occupation (label)
                occupation = example.get("occupation", "unknown")

                # Extract features
                skills = processor.extract_skills(bio)
                experience = processor.extract_experience(bio)
                education = processor.extract_education(bio)
                domain = processor.extract_domain(bio)

                # Store demographics if available
                demographics = {}
                if "gender" in example:
                    demographics["gender"] = example["gender"]
                if "name" in example:
                    demographics["name"] = example["name"]

                resume = Resume(
                    skill_tokens=skills,
                    years_experience=experience,
                    education_level=education,
                    domain_background=domain,
                    demographics=demographics
                )

                resumes.append(resume)
                labels.append(occupation)

            except Exception as e:
                print(f"Warning: Error processing Bias-in-Bios example: {e}")
                continue

        print(f"Loaded {len(resumes)} examples from Bias-in-Bios")
        return resumes, labels, processor.vocabulary

    except ImportError:
        print("datasets library required for Bias-in-Bios. Install with: pip install datasets")
        return None
    except Exception as e:
        print(f"Error loading Bias-in-Bios: {e}")
        return None
