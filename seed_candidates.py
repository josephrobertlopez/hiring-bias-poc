import pandas as pd, requests, json, random

df = pd.read_csv("data/raw/resume_dataset/Resume.csv")
samples = df.sample(10, random_state=42)

for _, row in samples.iterrows():
    text = str(row.get("Resume_str", ""))
    words = [w.lower() for w in text.split() if len(w) > 4][:10]
    category = str(row.get("Category", "tech")).lower().replace(" ", "_")
    payload = {
        "skill_tokens": words if words else ["python", "sql"],
        "years_experience": round(random.uniform(1, 10), 1),
        "education_level": random.choice(["bachelor", "master", "phd"]),
        "domain_background": [category],
        "demographics": {"gender": random.randint(0, 1)}
    }
    r = requests.post("http://localhost:5000/upload_resume", data={"resume": json.dumps(payload)})
    data = r.json()
    print(f"Candidate {data.get('candidate_id')} | score={data.get('confidence_score', '?'):.2f} | {category}")
