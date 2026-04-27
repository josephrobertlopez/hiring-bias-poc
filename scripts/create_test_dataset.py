"""Create synthetic resume dataset for testing validation pipeline."""
import csv
from pathlib import Path
import random

# Sample resume templates
RESUME_TEMPLATES = [
    # Software Engineer roles
    {
        "category": "Software Engineer",
        "skills": ["python", "java", "javascript", "sql", "aws", "docker", "kubernetes"],
        "exp_range": (3, 8),
        "edu": ["bachelor", "master"],
        "text": "Software Engineer with {years} years of experience. Proficient in {skills}. "
               "Experienced with cloud platforms and container orchestration."
    },
    # Data Scientist roles
    {
        "category": "Data Scientist",
        "skills": ["python", "r", "sql", "tensorflow", "sklearn", "pandas", "tableau"],
        "exp_range": (2, 7),
        "edu": ["master", "phd"],
        "text": "Data Scientist with {years} years of experience in machine learning. "
               "Strong background in {skills}. Published research on predictive analytics."
    },
    # DevOps Engineer
    {
        "category": "DevOps Engineer",
        "skills": ["docker", "kubernetes", "aws", "terraform", "jenkins", "linux", "bash"],
        "exp_range": (3, 9),
        "edu": ["bachelor", "bootcamp"],
        "text": "DevOps Engineer with {years} years of experience. Expert in {skills}. "
               "Managed infrastructure for high-traffic applications."
    },
    # Data Engineer
    {
        "category": "Data Engineer",
        "skills": ["python", "sql", "spark", "hadoop", "aws", "kafka"],
        "exp_range": (2, 8),
        "edu": ["bachelor", "master"],
        "text": "Data Engineer with {years} years of experience. Proficient in {skills}. "
               "Built scalable data pipelines and ETL systems."
    },
    # Frontend Engineer
    {
        "category": "Frontend Engineer",
        "skills": ["javascript", "react", "html", "css", "typescript", "node.js"],
        "exp_range": (2, 6),
        "edu": ["bachelor", "bootcamp"],
        "text": "Frontend Engineer with {years} years of experience. Skilled in {skills}. "
               "Built responsive and accessible user interfaces."
    },
    # Backend Engineer
    {
        "category": "Backend Engineer",
        "skills": ["python", "java", "sql", "rest api", "microservices", "aws"],
        "exp_range": (3, 8),
        "edu": ["bachelor", "master"],
        "text": "Backend Engineer with {years} years of experience. Expert in {skills}. "
               "Designed and implemented scalable backend systems."
    },
    # Security Engineer
    {
        "category": "Security Engineer",
        "skills": ["linux", "python", "networking", "cryptography", "penetration testing"],
        "exp_range": (4, 10),
        "edu": ["master", "phd"],
        "text": "Security Engineer with {years} years of experience. Specialist in {skills}. "
               "Conducted security audits and implemented security protocols."
    },
    # QA Engineer
    {
        "category": "QA Engineer",
        "skills": ["selenium", "pytest", "java", "sql", "automation", "jira"],
        "exp_range": (2, 6),
        "edu": ["bachelor", "bootcamp"],
        "text": "QA Engineer with {years} years of experience. Proficient in {skills}. "
               "Implemented automated testing frameworks and test strategies."
    },
    # Product Manager
    {
        "category": "Product Manager",
        "skills": ["product management", "data analysis", "communication", "leadership", "sql"],
        "exp_range": (3, 10),
        "edu": ["bachelor", "master", "mba"],
        "text": "Product Manager with {years} years of experience. Strong in {skills}. "
               "Led cross-functional teams and launched successful products."
    },
    # Business Analyst
    {
        "category": "Business Analyst",
        "skills": ["sql", "data analysis", "communication", "project management", "excel"],
        "exp_range": (2, 7),
        "edu": ["bachelor", "master"],
        "text": "Business Analyst with {years} years of experience. Skilled in {skills}. "
               "Gathered requirements and improved business processes."
    }
]

def create_synthetic_dataset(output_path: str = "data/raw/resume_dataset/Resume.csv", num_samples: int = 200):
    """Create synthetic resume dataset."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # Generate samples for each category
    samples_per_category = num_samples // len(RESUME_TEMPLATES)

    for template in RESUME_TEMPLATES:
        for _ in range(samples_per_category):
            # Randomize details
            years = random.randint(*template["exp_range"])
            selected_skills = random.sample(template["skills"], k=random.randint(3, len(template["skills"])))
            edu = random.choice(template["edu"])

            # Build resume text
            resume_text = template["text"].format(
                years=years,
                skills=", ".join(selected_skills)
            )

            # Add education info
            if edu == "master":
                resume_text += " Master's degree in Computer Science."
            elif edu == "phd":
                resume_text += " PhD in related field with advanced research."
            elif edu == "bootcamp":
                resume_text += " Completed coding bootcamp certification."
            else:
                resume_text += " Bachelor's degree in Computer Science or related field."

            # Add some random skills to the text
            available_extra = [s for s in RESUME_TEMPLATES[0]["skills"] if s not in selected_skills]
            if available_extra:
                num_extra = min(random.randint(0, 3), len(available_extra))
                if num_extra > 0:
                    extra_skills = random.sample(available_extra, k=num_extra)
                    resume_text += f" Additional skills: {', '.join(extra_skills)}."

            rows.append({
                "Resume_str": resume_text,
                "Category": template["category"]
            })

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Resume_str", "Category"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Created synthetic dataset: {output_path}")
    print(f"  Samples: {len(rows)}")
    print(f"  Categories: {len(set(r['Category'] for r in rows))}")

    return output_path

if __name__ == "__main__":
    create_synthetic_dataset()
