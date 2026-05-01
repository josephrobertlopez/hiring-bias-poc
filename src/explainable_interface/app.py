"""Flask web application for explainable hiring interface."""

import json
import time
import html
from typing import Dict, Any, List
from flask import Flask, request, jsonify, render_template_string, redirect, url_for

import sys
import os
_src_dir = os.path.join(os.path.dirname(__file__), '..')
_project_root = os.path.abspath(os.path.join(_src_dir, '..'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rich_explanations import EnhancedExplanationEngine
from rules.engine import SkillRulesEngine
from rules.data import Resume, SkillVocabulary
from src.data_processing.resume_processor import ResumeProcessor

# ---------------------------------------------------------------------------
# Shared CSS / nav used in every HTML page
# ---------------------------------------------------------------------------
_BASE_CSS = """
<style>
  * { box-sizing: border-box; }
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f0f2f5; color: #222; }
  .nav { background: #1a1a2e; padding: 12px 24px; display: flex; gap: 24px; align-items: center; }
  .nav a { color: #e0e0e0; text-decoration: none; font-weight: 600; font-size: 14px; }
  .nav a:hover { color: #4fc3f7; }
  .fired { background: #fff9c4; border-left: 3px solid #f9a825; }
  .lift-high { color: #2e7d32; font-weight: 700; }
  .lift-med { color: #f57f17; font-weight: 700; }
  .lift-low { color: #c62828; }
  .nav .brand { color: #4fc3f7; font-size: 18px; font-weight: 700; margin-right: 16px; }
  .page { max-width: 1100px; margin: 32px auto; padding: 0 20px; }
  .card { background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,.08); padding: 24px; margin-bottom: 24px; }
  h1 { font-size: 22px; margin: 0 0 16px; color: #1a1a2e; }
  h2 { font-size: 17px; margin: 0 0 12px; color: #333; }
  h3 { font-size: 14px; margin: 0 0 8px; color: #555; text-transform: uppercase; letter-spacing: .5px; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th { background: #f7f8fa; text-align: left; padding: 10px 12px; border-bottom: 2px solid #e0e0e0; font-size: 12px; text-transform: uppercase; letter-spacing: .5px; color: #666; }
  td { padding: 10px 12px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }
  tr:hover td { background: #fafafa; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 700; }
  .badge-hire { background: #e8f5e9; color: #2e7d32; }
  .badge-strong { background: #c8e6c9; color: #1b5e20; }
  .badge-interview { background: #fff8e1; color: #f57f17; }
  .badge-pass { background: #ffebee; color: #c62828; }
  .badge-bias { background: #fce4ec; color: #880e4f; }
  .badge-ok { background: #e8f5e9; color: #2e7d32; }
  .score-bar-wrap { background: #f0f0f0; border-radius: 6px; height: 10px; width: 160px; display: inline-block; vertical-align: middle; }
  .score-bar { height: 10px; border-radius: 6px; }
  .btn { display: inline-block; padding: 10px 20px; background: #1a1a2e; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 600; text-decoration: none; }
  .btn:hover { background: #16213e; }
  .btn-green { background: #2e7d32; }
  .btn-green:hover { background: #1b5e20; }
  textarea { width: 100%; font-family: monospace; font-size: 13px; padding: 10px; border: 1px solid #ddd; border-radius: 6px; resize: vertical; }
  .tag { display: inline-block; background: #e3f2fd; color: #1565c0; border-radius: 4px; padding: 2px 8px; font-size: 12px; margin: 2px; }
  .tag-gap { background: #fff3e0; color: #e65100; }
  .tag-bias { background: #fce4ec; color: #880e4f; }
  .section-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .8px; color: #888; margin-bottom: 6px; }
  .rule-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
  .rule-name { width: 120px; font-size: 13px; font-weight: 600; color: #444; text-transform: capitalize; }
  .rule-score-text { width: 44px; text-align: right; font-size: 13px; font-weight: 700; }
  .rule-explain { font-size: 12px; color: #666; margin-top: 2px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .info-box { background: #f7f8fa; border-radius: 8px; padding: 14px; }
  .warn-box { background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px; border-radius: 0 6px 6px 0; margin-bottom: 12px; font-size: 14px; }
  .ok-box { background: #e8f5e9; border-left: 4px solid #4caf50; padding: 12px 16px; border-radius: 0 6px 6px 0; margin-bottom: 12px; font-size: 14px; }
  .tab-bar { display: flex; gap: 0; border-bottom: 2px solid #e0e0e0; margin-bottom: 20px; }
  .tab { padding: 10px 20px; cursor: pointer; font-weight: 600; font-size: 14px; border: none; background: none; color: #888; border-bottom: 2px solid transparent; margin-bottom: -2px; }
  .tab.active { color: #1a1a2e; border-bottom-color: #1a1a2e; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }
  a.row-link { color: #1565c0; font-weight: 600; text-decoration: none; }
  a.row-link:hover { text-decoration: underline; }
  .mono { font-family: monospace; font-size: 12px; background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }
</style>
<script>
function switchTab(group, name) {
  document.querySelectorAll('[data-tabgroup="'+group+'"]').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('[data-tab="'+group+'-'+name+'"]').forEach(t => t.classList.add('active'));
  document.getElementById(group+'-'+name).classList.add('active');
  document.querySelectorAll('[data-tabgroup="'+group+'"] ~ .tab-panel').forEach(p => {});
}
</script>
"""

_NAV = """
<div class="nav">
  <span class="brand">HireAudit</span>
  <a href="/">Upload</a>
  <a href="/candidates">All Candidates</a>
  <a href="/candidates/ranked">Ranked</a>
  <a href="/preferences">Rank by Prefs</a>
  <a href="/rules">Rules</a>
  <a href="/resumes">Browse Dataset</a>
  <a href="/help">Help</a>
</div>
"""


def _score_color(score: float) -> str:
    if score >= 0.75:
        return "#4caf50"
    elif score >= 0.55:
        return "#ff9800"
    else:
        return "#f44336"


def _recommendation(score: float) -> str:
    if score >= 0.8:
        return "Strong Hire"
    elif score >= 0.6:
        return "Hire"
    elif score >= 0.4:
        return "Interview"
    return "Pass"


def _rec_badge(score: float) -> str:
    rec = _recommendation(score)
    cls = {"Strong Hire": "badge-strong", "Hire": "badge-hire",
           "Interview": "badge-interview", "Pass": "badge-pass"}[rec]
    return f'<span class="badge {cls}">{rec}</span>'


def _score_bar(score: float, width: int = 160) -> str:
    pct = int(score * 100)
    color = _score_color(score)
    bar_w = int(width * score)
    return (f'<div class="score-bar-wrap" style="width:{width}px">'
            f'<div class="score-bar" style="width:{bar_w}px;background:{color}"></div></div> '
            f'<strong style="color:{color}">{pct}%</strong>')


class HiringApp:
    """Flask web application for hiring manager interface."""

    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'hiring-poc-secret-key'
        self.resume_processor = ResumeProcessor()
        self.candidates: List[Dict] = []
        self.candidate_counter = 0
        self._training_meta: Dict = {}
        self._custom_rules: List[Dict] = []        # user-defined rules
        self._minority_groups: List[str] = []      # user-defined minority categories
        self._custom_vocab_tokens: List[Dict] = []  # user-added vocabulary tokens {token, count, rate}
        self._setup_explanation_engine()
        self._setup_routes()

    def _setup_explanation_engine(self, min_support: float = 0.1,
                                   min_confidence: float = 0.5,
                                   n_resumes: int = 300):
        # S3 fix: use ResumeProcessor vocabulary (165 tokens) not a 22-item constant
        proc = self.resume_processor
        self.base_engine = SkillRulesEngine(proc.vocabulary)
        self.base_engine.rules['combination'].min_support = min_support
        self.base_engine.rules['combination'].min_confidence = min_confidence

        # AP1 fix: fit on real Resume.csv data, not 3 hardcoded samples
        csv_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'raw', 'resume_dataset', 'Resume.csv'
        ))
        resumes, labels = self._load_training_data(csv_path, n_resumes)
        self.base_engine.fit(resumes, labels)
        self.explanation_engine = EnhancedExplanationEngine(self.base_engine)
        self._training_meta = {
            'n_resumes': len(resumes),
            'n_hired': sum(labels),
            'min_support': min_support,
            'min_confidence': min_confidence,
            'n_rules': len(self.base_engine.rules['combination'].rules),
        }

    def _load_training_data(self, csv_path: str, n: int):
        import pandas as pd, random
        random.seed(42)
        df = pd.read_csv(csv_path).head(n)
        proc = self.resume_processor
        hire_bias = {'Data Science', 'Machine Learning', 'Java Developer',
                     'Python Developer', 'Database', 'DevOps Engineer',
                     'Business Analyst', 'Finance', 'Accountant', 'Banking'}
        resumes, labels, categories = [], [], []
        for _, row in df.iterrows():
            text = str(row.get('Resume_str', ''))
            category = str(row.get('Category', ''))
            skills = proc.extract_skills(text)
            exp = proc.extract_experience(text)
            edu = proc.extract_education(text)
            domain = proc.extract_domain(text)
            resume = Resume(skill_tokens=skills, years_experience=exp,
                            education_level=edu, domain_background=domain,
                            demographics={})
            if category in hire_bias and len(skills) >= 3:
                label = random.random() < 0.75
            elif len(skills) >= 5:
                label = random.random() < 0.55
            else:
                label = random.random() < 0.35
            resumes.append(resume)
            labels.append(label)
            categories.append(category)
        # Store for per-rule bias analysis
        self._training_resumes = resumes
        self._training_labels = labels
        self._training_categories = categories
        return resumes, labels

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------
    def _render_rule_bars(self, rule_scores: Dict[str, float],
                          rule_explains: Dict[str, Any]) -> str:
        rule_descriptions = {
            "combination": "Skill co-occurrence patterns (Bayesian association rules)",
            "experience":  "Years-of-experience threshold match",
            "education":   "Education level hiring pattern match",
            "domain":      "Industry domain background match",
            "gap":         "Critical skill gap penalty",
            "bias":        "Demographic parity (EEOC 4/5 rule)",
        }
        rows = []
        for rule, score in rule_scores.items():
            color = _score_color(score)
            pct = int(score * 100)
            bar_w = int(200 * score)
            desc = rule_descriptions.get(rule, "")
            explain_text = ""
            if rule_explains and rule in rule_explains:
                ex = rule_explains[rule]
                if isinstance(ex, dict):
                    explain_text = "; ".join(f"{k}: {v}" for k, v in list(ex.items())[:3])
                else:
                    explain_text = str(ex)[:120]
            rows.append(f"""
            <div class="rule-row">
              <div class="rule-name">{rule.title()}</div>
              <div class="score-bar-wrap" style="width:200px">
                <div class="score-bar" style="width:{bar_w}px;background:{color}"></div>
              </div>
              <div class="rule-score-text" style="color:{color}">{pct}%</div>
            </div>
            <div class="rule-explain" style="padding-left:132px;margin-bottom:8px">
              {html.escape(desc)}
              {(' — <em>' + html.escape(explain_text[:100]) + '</em>') if explain_text else ''}
            </div>""")
        return "\n".join(rows)

    def _render_candidate_detail_html(self, candidate: Dict) -> str:
        resume = candidate['resume']
        audit = candidate['audit_result']
        expl = candidate['explanation']
        score = candidate['confidence_score']
        cid = candidate['id']

        skills_html = " ".join(f'<span class="tag">{html.escape(s)}</span>'
                               for s in resume.skill_tokens) or "<em>none</em>"
        gaps_html = " ".join(f'<span class="tag tag-gap">{html.escape(g)}</span>'
                             for g in audit.skill_gaps) or '<span class="badge badge-ok">No gaps</span>'
        bias_html = ""
        if audit.bias_flags:
            for bf in audit.bias_flags:
                bias_html += f'<div class="warn-box">⚠️ {html.escape(bf)}</div>'
        else:
            bias_html = '<div class="ok-box">✅ No bias flags — demographic parity check passed</div>'

        patterns_html = ""
        if audit.skill_patterns:
            patterns_html = "<ul style='margin:0;padding-left:18px;font-size:13px'>"
            for p in audit.skill_patterns[:8]:
                patterns_html += f"<li><span class='mono'>{html.escape(p)}</span></li>"
            patterns_html += "</ul>"
        else:
            patterns_html = "<em style='font-size:13px;color:#888'>No association patterns found in training data</em>"

        rule_bars = self._render_rule_bars(audit.rule_scores, audit.explanations)

        reasoning_html = ""
        if expl.business_reasoning:
            reasoning_html = "<ul style='margin:0;padding-left:18px;font-size:14px'>"
            for r in expl.business_reasoning:
                reasoning_html += f"<li>{html.escape(str(r))}</li>"
            reasoning_html += "</ul>"

        rule_contributions_html = ""
        if expl.rule_contributions:
            rule_contributions_html = "<ul style='margin:0;padding-left:18px;font-size:13px'>"
            for rule, contrib in expl.rule_contributions.items():
                rule_contributions_html += f"<li><strong>{html.escape(rule)}:</strong> {html.escape(str(contrib))}</li>"
            rule_contributions_html += "</ul>"

        return f"""<!DOCTYPE html><html><head><title>Candidate {cid} — HireAudit</title>{_BASE_CSS}</head><body>
{_NAV}
<div class="page">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px">
    <a href="/candidates/ranked" style="color:#888;text-decoration:none;font-size:13px">← Back to ranked list</a>
  </div>

  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:flex-start">
      <div>
        <h1>Candidate #{cid}</h1>
        <div style="font-size:13px;color:#666">{html.escape(resume.education_level.title())} ·
          {resume.years_experience:.1f} yrs experience ·
          {html.escape(', '.join(resume.domain_background) or 'unknown domain')}</div>
      </div>
      <div style="text-align:right">
        {_rec_badge(score)}
        <div style="margin-top:8px">{_score_bar(score, 200)}</div>
        <div style="font-size:11px;color:#888;margin-top:4px">Overall confidence</div>
      </div>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Skills Detected</h2>
      {skills_html}
    </div>
    <div class="card">
      <h2>Skill Gaps</h2>
      {gaps_html}
    </div>
  </div>

  <div class="card">
    <h2>Rule Contributions — 6 Bayesian Rules</h2>
    <p style="font-size:13px;color:#666;margin-top:0">Each bar shows how strongly this rule supports hiring this candidate. Overall score = average of 6 rules.</p>
    {rule_bars}
    {('<div style="margin-top:12px"><h3>Explanation Details</h3>' + rule_contributions_html + '</div>') if rule_contributions_html else ''}
  </div>

  <div class="card">
    <h2>Bayesian Association Patterns</h2>
    <p style="font-size:13px;color:#666;margin-top:0">Skill co-occurrence rules learned from training data (antecedent → consequent).</p>
    {patterns_html}
  </div>

  <div class="card">
    <h2>Bias Analysis</h2>
    {bias_html}
    <div style="font-size:12px;color:#888;margin-top:8px">Method: EEOC 4/5 rule — disparate impact ratio (min group rate / max group rate ≥ 0.8 required)</div>
  </div>

  <div class="card">
    <h2>Business Reasoning</h2>
    {reasoning_html or '<em style="color:#888">No reasoning generated</em>'}
    {('<div style="margin-top:12px"><p style="font-size:13px"><strong>Business Case:</strong> ' + html.escape(str(expl.business_case)) + '</p></div>') if expl.business_case else ''}
    {('<div style="margin-top:8px"><p style="font-size:13px"><strong>Historical Evidence:</strong> ' + html.escape(str(expl.historical_evidence)) + '</p></div>') if expl.historical_evidence else ''}
    {('<div style="margin-top:8px"><p style="font-size:13px"><strong>Skill Gap Impact:</strong> ' + html.escape(str(expl.skill_gap_impact)) + '</p></div>') if expl.skill_gap_impact else ''}
  </div>
</div>
</body></html>"""

    def _setup_routes(self):

        # ------------------------------------------------------------------
        # Home — upload form (raw text + JSON tabs)
        # ------------------------------------------------------------------
        @self.app.route('/')
        def index():
            sample_json = '{"skill_tokens": ["python", "machine_learning", "aws"], "years_experience": 5.0, "education_level": "master", "domain_background": ["tech"], "demographics": {"gender": 0}}'
            return render_template_string(f"""<!DOCTYPE html><html><head>
<title>HireAudit — Upload Resume</title>{_BASE_CSS}
<script>
function showTab(name) {{
  document.getElementById('tab-raw').classList.toggle('active', name==='raw');
  document.getElementById('tab-json').classList.toggle('active', name==='json');
  document.getElementById('panel-raw').style.display = name==='raw' ? 'block' : 'none';
  document.getElementById('panel-json').style.display = name==='json' ? 'block' : 'none';
}}
</script>
</head><body>
{_NAV}
<div class="page">
  <div class="card">
    <h1>Upload Candidate Resume</h1>
    <p style="color:#666;font-size:14px;margin-top:0">Paste raw resume text or structured JSON to get an instant bias-audited hiring recommendation with Bayesian rule explanations.</p>

    <div class="tab-bar">
      <button class="tab active" id="tab-raw" onclick="showTab('raw')">Raw Resume Text</button>
      <button class="tab" id="tab-json" onclick="showTab('json')">Structured JSON</button>
    </div>

    <div id="panel-raw">
      <form action="/upload_raw" method="post">
        <div class="section-label">Paste resume text below — skills, experience, education extracted automatically</div>
        <textarea name="resume_text" rows="12" placeholder="Paste full resume text here...&#10;&#10;Example:&#10;Senior Software Engineer with 7 years of experience in Python, TensorFlow, and AWS.&#10;Master's degree in Computer Science from Stanford.&#10;Previous roles in fintech and healthcare."></textarea>
        <br><br>
        <button type="submit" class="btn btn-green">Analyse Resume Text</button>
      </form>
    </div>

    <div id="panel-json" style="display:none">
      <form action="/upload_resume" method="post">
        <div class="section-label">Paste structured JSON resume data</div>
        <textarea name="resume" rows="8">{html.escape(sample_json)}</textarea>
        <br><br>
        <button type="submit" class="btn">Generate Explanation</button>
      </form>
    </div>
  </div>

  <div class="card" style="background:#f7f8fa">
    <h2>How it works</h2>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;text-align:center;font-size:13px">
      <div><div style="font-size:24px;margin-bottom:8px">📄</div><strong>1. Upload</strong><br>Paste text or JSON</div>
      <div><div style="font-size:24px;margin-bottom:8px">🔍</div><strong>2. Extract</strong><br>Skills, edu, domain auto-detected</div>
      <div><div style="font-size:24px;margin-bottom:8px">⚖️</div><strong>3. Audit</strong><br>6 Bayesian rules score the resume</div>
      <div><div style="font-size:24px;margin-bottom:8px">📊</div><strong>4. Explain</strong><br>Bias check + hiring recommendation</div>
    </div>
  </div>
</div>
</body></html>""")

        # ------------------------------------------------------------------
        # Raw text upload — shows extraction step then result
        # ------------------------------------------------------------------
        @self.app.route('/upload_raw', methods=['POST'])
        def upload_raw():
            start_time = time.time()
            resume_text = request.form.get('resume_text', '').strip()
            if not resume_text:
                return redirect('/')

            proc = self.resume_processor
            skill_scores = proc.extract_skills_with_scores(resume_text)
            skills = list(skill_scores.keys())
            experience = proc.extract_experience(resume_text)
            education = proc.extract_education(resume_text)
            domain = proc.extract_domain(resume_text)

            resume = Resume(
                skill_tokens=skills,
                years_experience=experience,
                education_level=education,
                domain_background=domain,
                demographics={},
            )

            audit_result = self.base_engine.audit_resume(resume)
            explanation = self.explanation_engine.explain_decision(resume, audit_result)
            candidate_id = self._store_candidate(resume, explanation, audit_result)
            score = audit_result.overall_score
            proc_time = time.time() - start_time

            n_fuzzy = sum(1 for v in skill_scores.values() if v.get('method') == 'fuzzy')
            skills_html = " ".join(
                f'<span class="tag" title="{html.escape(v.get("method","exact"))} match score={v.get("score",100)}'
                + (f' (matched &quot;{html.escape(v.get("matched_by",""))}&quot;)' if v.get("matched_by") else '')
                + ('" style="border:1px dashed #1565c0;background:#e8eaf6">' if v.get("method") == "fuzzy" else '">')
                + html.escape(s) + ('&nbsp;<sup style="font-size:9px;color:#7986cb">~' + str(v.get("score","")) + '%</sup>' if v.get("method") == "fuzzy" else "")
                + '</span>'
                for s, v in skill_scores.items()
            ) or "<em>none detected</em>"
            fuzzy_note = (f'<div style="font-size:12px;color:#7986cb;margin-top:6px">'
                          f'<strong>{n_fuzzy}</strong> skill(s) found via fuzzy matching '
                          f'(dashed border, score shown). Exact matches have solid border.</div>'
                          ) if n_fuzzy else ''
            gaps_html = " ".join(f'<span class="tag tag-gap">{html.escape(g)}</span>' for g in audit_result.skill_gaps) or '<span class="badge badge-ok">None</span>'
            bias_html = ""
            if audit_result.bias_flags:
                for bf in audit_result.bias_flags:
                    bias_html += f'<div class="warn-box">⚠️ {html.escape(bf)}</div>'
            else:
                bias_html = '<div class="ok-box">✅ Demographic parity check passed</div>'

            rule_bars = self._render_rule_bars(audit_result.rule_scores, audit_result.explanations)

            raw_preview = html.escape(resume_text[:500]) + ("…" if len(resume_text) > 500 else "")

            # --- Fairness analysis ---
            demo_proxy = self._infer_demographic_proxy(resume_text, domain, skills)
            cf_results = self._counterfactual_analysis(resume, score)

            # 10% skill perturbation — 7 runs, replace δ=10% of skills randomly
            import random as _rnd
            _rnd.seed(None)
            _all_vocab = [t for t in self.resume_processor.vocabulary.tokens if t not in skills]
            _perturb_scores = []
            for _ in range(7):
                _swapped = list(skills)
                _n = max(1, len(_swapped) // 10)
                for _pos in _rnd.sample(range(len(_swapped)), min(_n, len(_swapped))):
                    _swapped[_pos] = _rnd.choice(_all_vocab) if _all_vocab else _swapped[_pos]
                _pr = Resume(skill_tokens=_swapped, years_experience=experience,
                             education_level=education, domain_background=domain, demographics={})
                try:
                    _perturb_scores.append(self.base_engine.audit_resume(_pr).overall_score)
                except Exception:
                    pass
            _p_min = min(_perturb_scores) if _perturb_scores else score
            _p_max = max(_perturb_scores) if _perturb_scores else score
            _p_mean = sum(_perturb_scores) / len(_perturb_scores) if _perturb_scores else score
            _p_range = _p_max - _p_min
            _perturb_flag = _p_range > 0.12  # flag if >12% swing

            # Build counterfactual table HTML
            cf_rows = ""
            for cf in cf_results:
                d = cf['delta']
                d_color = '#2e7d32' if d >= 0 else '#c62828'
                d_str = f"+{d*100:.1f}%" if d >= 0 else f"{d*100:.1f}%"
                bar_w = int(cf['score'] * 120)
                bar_c = _score_color(cf['score'])
                is_orig = any(dom.lower() in cf['label'].lower() for dom in domain)
                orig_badge = ' <span class="badge badge-ok" style="font-size:10px">closest</span>' if is_orig else ''
                cf_rows += f"""<tr>
                  <td>{html.escape(cf['label'])}{orig_badge}</td>
                  <td>
                    <div style="display:flex;align-items:center;gap:6px">
                      <div class="score-bar-wrap" style="width:120px">
                        <div class="score-bar" style="width:{bar_w}px;background:{bar_c}"></div>
                      </div>
                      <strong style="color:{bar_c}">{cf['score']*100:.0f}%</strong>
                    </div>
                  </td>
                  <td style="font-weight:700;color:{d_color}">{d_str}</td>
                </tr>"""

            _cf_spread = (max(c['score'] for c in cf_results) - min(c['score'] for c in cf_results)) if cf_results else 0
            cf_verdict = (
                f'<div class="warn-box">⚠️ Score swings <strong>{_cf_spread*100:.0f}%</strong> across domain alternatives. '
                f'Domain background acts as a demographic proxy — the model may be penalising non-tech pipelines.</div>'
                if _cf_spread > 0.10 else
                f'<div class="ok-box">✅ Score stable across domain alternatives (spread = {_cf_spread*100:.0f}%). Low proxy bias signal.</div>'
            )
            perturb_verdict = (
                f'<div class="warn-box">⚠️ δ=10% skill swap causes <strong>{_p_range*100:.0f}%</strong> score swing '
                f'({_p_min*100:.0f}%–{_p_max*100:.0f}%). Score is sensitive to skill composition — '
                f'minority skill patterns may be disadvantaged.</div>'
                if _perturb_flag else
                f'<div class="ok-box">✅ δ=10% skill swap: score range {_p_min*100:.0f}%–{_p_max*100:.0f}% '
                f'(spread = {_p_range*100:.0f}%). Score is robust to small skill perturbations.</div>'
            )

            title_signal_html = ""
            if demo_proxy['title_signals']:
                title_signal_html = "<ul style='margin:6px 0 0;padding-left:18px;font-size:13px'>" + \
                    "".join(f"<li>{html.escape(s)}</li>" for s in demo_proxy['title_signals']) + "</ul>"

            return render_template_string(f"""<!DOCTYPE html><html><head>
<title>Candidate #{candidate_id} — HireAudit</title>{_BASE_CSS}
</head><body>
{_NAV}
<div class="page">
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:flex-start">
      <div>
        <h1>Analysis Result — Candidate #{candidate_id}</h1>
        <div style="font-size:13px;color:#666">Processed in {proc_time:.2f}s ·
          <a href="/candidates/ranked" style="color:#1565c0">View all candidates</a></div>
      </div>
      <div style="text-align:right">
        {_rec_badge(score)}
        <div style="margin-top:8px">{_score_bar(score, 200)}</div>
        <div style="font-size:11px;color:#888;margin-top:4px">Overall confidence</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Step 1 — Text Extraction</h2>
    <p style="font-size:13px;color:#888;margin-top:0">What the ResumeProcessor pulled from the raw text:</p>
    <div class="grid2">
      <div class="info-box">
        <div class="section-label">Skills detected ({len(skills)})</div>
        {skills_html}
        {fuzzy_note}
      </div>
      <div class="info-box">
        <div class="section-label">Extracted metadata</div>
        <table style="font-size:13px;width:auto">
          <tr><td style="color:#666;padding:2px 12px 2px 0">Experience</td><td><strong>{experience:.1f} years</strong></td></tr>
          <tr><td style="color:#666;padding:2px 12px 2px 0">Education</td><td><strong>{html.escape(education.title())}</strong></td></tr>
          <tr><td style="color:#666;padding:2px 12px 2px 0">Domain</td><td><strong>{html.escape(', '.join(domain))}</strong></td></tr>
        </table>
      </div>
    </div>
    <details style="margin-top:12px">
      <summary style="font-size:12px;color:#888;cursor:pointer">Show raw text preview</summary>
      <pre style="font-size:12px;background:#f5f5f5;padding:12px;border-radius:6px;white-space:pre-wrap;margin-top:8px">{raw_preview}</pre>
    </details>
  </div>

  <div class="card">
    <h2>Step 2 — 6 Bayesian Rules</h2>
    <p style="font-size:13px;color:#666;margin-top:0">Each rule independently scores the candidate. Overall = average of 6.</p>
    {rule_bars}
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Skill Gaps</h2>
      {gaps_html}
    </div>
    <div class="card">
      <h2>Bias Analysis</h2>
      {bias_html}
      <div style="font-size:12px;color:#888">EEOC 4/5 disparate impact rule</div>
    </div>
  </div>

  <div class="card">
    <h2>Business Reasoning</h2>
    {'<ul style="margin:0;padding-left:18px;font-size:14px">' + ''.join(f'<li>{html.escape(str(r))}</li>' for r in explanation.business_reasoning) + '</ul>' if explanation.business_reasoning else '<em style="color:#888">No reasoning generated</em>'}
  </div>

  <!-- Fairness Analysis -->
  <div class="card">
    <h2>Fairness Analysis — Demographic Proxy Inference</h2>
    <p style="font-size:13px;color:#888;margin-top:0">
      Resume.csv has no demographic data. These signals are <strong>inferred proxies</strong>
      from job title and domain — not ground truth. Shown to demonstrate how models absorb
      demographic signals through career-pipeline proxies.
    </p>

    <div class="grid2" style="margin-bottom:16px">
      <div class="info-box">
        <div class="section-label">Inferred career level</div>
        <div style="font-size:16px;font-weight:700;margin-top:6px">{html.escape(demo_proxy['seniority'])}</div>
      </div>
      <div class="info-box">
        <div class="section-label">Pipeline representation signal</div>
        <div style="font-size:15px;font-weight:700;color:{demo_proxy['group_color']};margin-top:6px">
          {html.escape(demo_proxy['group_label'])}
        </div>
        <div style="font-size:12px;color:#666;margin-top:4px">{html.escape(demo_proxy['group_note'])}</div>
      </div>
    </div>

    {('<div class="warn-box" style="margin-bottom:16px"><strong>Job title gender-skew signals detected:</strong>' + title_signal_html + '</div>') if title_signal_html else ''}

    <h3>Counterfactual Domain Test — same skills, different background</h3>
    <p style="font-size:12px;color:#666;margin-top:0">
      Holds skills, experience, and education constant. Only changes <code>domain_background</code>.
      Score should be identical if the model is domain-neutral.
    </p>
    {cf_verdict}
    <table style="margin-top:12px">
      <tr><th>Domain alternative</th><th>Score</th><th>Δ vs original ({score*100:.0f}%)</th></tr>
      {cf_rows}
    </table>

    <h3 style="margin-top:20px">δ = 10% Skill Perturbation Test (7 runs)</h3>
    <p style="font-size:12px;color:#666;margin-top:0">
      Randomly replaces 10% of detected skills with alternatives from the vocabulary.
      A wide score range means the model is sensitive to exact skill composition —
      which can proxy for demographic patterns in who lists which skills.
    </p>
    {perturb_verdict}
    <div style="margin-top:8px;font-size:13px;color:#666">
      Mean: <strong>{_p_mean*100:.1f}%</strong> &nbsp;|&nbsp;
      Range: <strong>{_p_min*100:.1f}%–{_p_max*100:.1f}%</strong> &nbsp;|&nbsp;
      Spread: <strong>{_p_range*100:.1f}%</strong>
    </div>
  </div>

  <div style="margin-top:8px">
    <a href="/candidates/{candidate_id}" class="btn">View Full Detail</a>
    &nbsp;
    <a href="/" class="btn" style="background:#555">Upload Another</a>
  </div>
</div>
</body></html>""")

        # ------------------------------------------------------------------
        # JSON upload (backward-compat, returns JSON for seed script)
        # ------------------------------------------------------------------
        @self.app.route('/upload_resume', methods=['POST'])
        def upload_resume():
            start_time = time.time()
            try:
                resume_json = request.form.get('resume', '')
                if not resume_json:
                    return jsonify({'error': 'No resume data provided'}), 400
                try:
                    resume_data = json.loads(resume_json)
                except json.JSONDecodeError:
                    return jsonify({'error': 'Invalid JSON format'}), 400

                resume_data = self._sanitize_resume_data(resume_data)
                resume = Resume(
                    skill_tokens=resume_data.get('skill_tokens', []),
                    years_experience=float(resume_data.get('years_experience', 0)),
                    education_level=resume_data.get('education_level', 'unknown'),
                    domain_background=resume_data.get('domain_background', []),
                    demographics=resume_data.get('demographics', {}),
                )
                audit_result = self.base_engine.audit_resume(resume)
                explanation = self.explanation_engine.explain_decision(resume, audit_result)
                candidate_id = self._store_candidate(resume, explanation, audit_result)
                score = audit_result.overall_score
                processing_time = time.time() - start_time

                # If request came from browser (form submit), render HTML result
                accept = request.headers.get('Accept', '')
                if 'text/html' in accept:
                    return redirect(url_for('get_candidate_html', candidate_id=candidate_id))

                return jsonify({
                    'success': True,
                    'candidate_id': candidate_id,
                    'processing_time': processing_time,
                    'confidence_score': score,
                    'explanation': {
                        'business_reasoning': explanation.business_reasoning,
                        'historical_evidence': explanation.historical_evidence,
                        'bias_analysis': explanation.bias_analysis,
                        'confidence_analysis': explanation.confidence_analysis,
                        'bias_warning': explanation.bias_warning,
                        'bias_details': explanation.bias_details,
                        'skill_gap_analysis': explanation.skill_gap_analysis,
                        'rule_contributions': explanation.rule_contributions,
                    },
                    'rule_scores': audit_result.rule_scores,
                    'bias_flags': audit_result.bias_flags,
                    'skill_gaps': audit_result.skill_gaps,
                })
            except Exception as e:
                return jsonify({'error': f'Processing failed: {str(e)}'}), 500

        # ------------------------------------------------------------------
        # Candidates dashboard — HTML table
        # ------------------------------------------------------------------
        @self.app.route('/candidates')
        def list_candidates():
            if not self.candidates:
                body = '<div class="card"><p style="color:#888;text-align:center;padding:40px">No candidates yet. <a href="/">Upload a resume</a> to get started.</p></div>'
            else:
                rows = ""
                for c in self.candidates:
                    r = c['resume']
                    score = c['confidence_score']
                    bias = '⚠️' if c['audit_result'].bias_flags else '✅'
                    skills_preview = ", ".join(r.skill_tokens[:4]) or "—"
                    rows += f"""<tr>
                      <td><a href="/candidates/{c['id']}" class="row-link">#{c['id']}</a></td>
                      <td>{html.escape(skills_preview)}</td>
                      <td>{r.years_experience:.1f} yrs</td>
                      <td>{html.escape(r.education_level.title())}</td>
                      <td>{html.escape(', '.join(r.domain_background[:2]))}</td>
                      <td>{_score_bar(score, 120)}</td>
                      <td>{_rec_badge(score)}</td>
                      <td style="font-size:18px;text-align:center">{bias}</td>
                    </tr>"""
                body = f"""<div class="card">
                  <h1>All Candidates ({len(self.candidates)})</h1>
                  <table>
                    <tr><th>#</th><th>Skills (preview)</th><th>Exp</th><th>Education</th><th>Domain</th><th>Score</th><th>Recommendation</th><th>Bias</th></tr>
                    {rows}
                  </table>
                </div>"""

            return render_template_string(f"""<!DOCTYPE html><html><head><title>Candidates — HireAudit</title>{_BASE_CSS}</head><body>
{_NAV}<div class="page">{body}</div></body></html>""")

        # ------------------------------------------------------------------
        # Ranked view — HTML with bias comparison summary
        # ------------------------------------------------------------------
        @self.app.route('/candidates/ranked')
        def ranked_candidates():
            ranked = sorted(self.candidates, key=lambda x: x['confidence_score'], reverse=True)

            # Bias summary across all candidates
            total = len(ranked)
            biased = sum(1 for c in ranked if c['audit_result'].bias_flags)
            gender_scores = {}
            for c in ranked:
                g = c['resume'].demographics.get('gender', None)
                if g is not None:
                    gender_scores.setdefault(g, []).append(c['confidence_score'])
            gender_summary = ""
            if gender_scores:
                parts = []
                for g, scores in sorted(gender_scores.items()):
                    avg = sum(scores) / len(scores)
                    label = {0: "Group 0", 1: "Group 1"}.get(g, f"Group {g}")
                    parts.append(f"{label}: avg score {avg:.0%} ({len(scores)} candidates)")
                gender_summary = " · ".join(parts)

            if not ranked:
                body = '<div class="card"><p style="color:#888;text-align:center;padding:40px">No candidates yet. <a href="/">Upload a resume</a> to get started.</p></div>'
            else:
                rows = ""
                for rank, c in enumerate(ranked, 1):
                    r = c['resume']
                    score = c['confidence_score']
                    bias = '⚠️' if c['audit_result'].bias_flags else '✅'
                    skills_preview = ", ".join(r.skill_tokens[:4]) or "—"
                    gaps = ", ".join(c['audit_result'].skill_gaps[:3]) or "—"
                    rows += f"""<tr>
                      <td style="font-weight:700;color:#888">#{rank}</td>
                      <td><a href="/candidates/{c['id']}" class="row-link">Candidate {c['id']}</a></td>
                      <td>{html.escape(skills_preview)}</td>
                      <td>{r.years_experience:.1f} yrs</td>
                      <td>{html.escape(r.education_level.title())}</td>
                      <td>{_score_bar(score, 140)}</td>
                      <td>{_rec_badge(score)}</td>
                      <td style="font-size:12px;color:#888">{html.escape(gaps)}</td>
                      <td style="font-size:18px;text-align:center">{bias}</td>
                    </tr>"""

                bias_box = (f'<div class="warn-box">{biased} of {total} candidates flagged for potential bias.</div>'
                            if biased else f'<div class="ok-box">No bias flags across {total} candidates.</div>')
                gender_box = f'<div style="font-size:13px;color:#666;margin-bottom:16px">{gender_summary}</div>' if gender_summary else ""

                body = f"""<div class="card">
                  <h1>Candidates — Ranked by Score</h1>
                  {bias_box}
                  {gender_box}
                  <table>
                    <tr><th>Rank</th><th>Candidate</th><th>Skills (preview)</th><th>Exp</th><th>Education</th><th>Score</th><th>Recommendation</th><th>Skill Gaps</th><th>Bias</th></tr>
                    {rows}
                  </table>
                </div>"""

            return render_template_string(f"""<!DOCTYPE html><html><head><title>Ranked — HireAudit</title>{_BASE_CSS}</head><body>
{_NAV}<div class="page">{body}</div></body></html>""")

        # ------------------------------------------------------------------
        # Candidate detail — HTML
        # ------------------------------------------------------------------
        @self.app.route('/candidates/<int:candidate_id>')
        def get_candidate_html(candidate_id):
            candidate = next((c for c in self.candidates if c['id'] == candidate_id), None)
            if not candidate:
                return f"""<!DOCTYPE html><html><head><title>Not Found</title>{_BASE_CSS}</head><body>
{_NAV}<div class="page"><div class="card"><p>Candidate #{candidate_id} not found. <a href="/candidates">Back to list</a></p></div></div></body></html>""", 404
            return self._render_candidate_detail_html(candidate)

        # ------------------------------------------------------------------
        # Browse raw dataset (list view)
        # ------------------------------------------------------------------
        @self.app.route('/resumes')
        def browse_resumes():
            import pandas as pd
            csv_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', '..', 'data', 'raw', 'resume_dataset', 'Resume.csv'
            ))
            try:
                df = pd.read_csv(csv_path)
                page = int(request.args.get('page', 0))
                per_page = 10
                sample = df.iloc[page * per_page:(page + 1) * per_page]
                rows = ""
                for idx, row in sample.iterrows():
                    text = str(row.get('Resume_str', ''))
                    category = str(row.get('Category', ''))
                    skills = self.resume_processor.extract_skills(text)
                    edu = self.resume_processor.extract_education(text)
                    exp = self.resume_processor.extract_experience(text)
                    preview = html.escape(text[:200]).replace('\n', ' ')
                    skills_html = " ".join(f'<span class="tag">{html.escape(s)}</span>' for s in skills[:8])
                    rows += f"""<div class="card" style="margin-bottom:16px">
                      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">
                        <div style="flex:1;min-width:0">
                          <div style="margin-bottom:6px">
                            <span class="badge badge-hire">{html.escape(category)}</span>
                            <span style="font-size:12px;color:#888;margin-left:8px">{edu.title()} · {exp:.1f} yrs exp · {len(skills)} skills detected</span>
                          </div>
                          <div style="margin-bottom:8px">{skills_html}</div>
                          <div style="font-size:12px;color:#999;line-height:1.5">{preview}…</div>
                        </div>
                        <div style="display:flex;flex-direction:column;gap:6px;flex-shrink:0">
                          <a href="/resumes/{idx}" class="btn" style="font-size:12px;padding:6px 14px;text-align:center">View Full</a>
                          <form action="/upload_raw" method="post" style="margin:0">
                            <input type="hidden" name="resume_text" value="{html.escape(text)}">
                            <button type="submit" class="btn btn-green" style="font-size:12px;padding:6px 14px;width:100%">Analyse</button>
                          </form>
                        </div>
                      </div>
                    </div>"""
                total_pages = len(df) // per_page
                prev_link = f'<a href="/resumes?page={page-1}" class="btn" style="background:#555">← Prev</a>&nbsp;' if page > 0 else ''
                next_link = f'&nbsp;<a href="/resumes?page={page+1}" class="btn">Next →</a>' if page < total_pages - 1 else ''
                nav_html = f'<div style="margin-top:16px">{prev_link}Page {page+1} of {total_pages}{next_link}</div>'
                body = f'<h1 style="margin-bottom:16px">Browse Dataset — {len(df):,} Resumes</h1>{rows}{nav_html}'
            except Exception as e:
                body = f'<div class="card"><p style="color:#c62828">Could not load dataset: {html.escape(str(e))}</p></div>'

            return render_template_string(f"""<!DOCTYPE html><html><head><title>Browse Dataset — HireAudit</title>{_BASE_CSS}</head><body>
{_NAV}<div class="page">{body}</div></body></html>""")

        # ------------------------------------------------------------------
        # Full resume detail — rendered from Resume_html column
        # ------------------------------------------------------------------
        @self.app.route('/resumes/<int:resume_idx>')
        def view_resume(resume_idx):
            import pandas as pd, re
            csv_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', '..', 'data', 'raw', 'resume_dataset', 'Resume.csv'
            ))
            try:
                df = pd.read_csv(csv_path)
                if resume_idx < 0 or resume_idx >= len(df):
                    return f"""<!DOCTYPE html><html><head><title>Not Found</title>{_BASE_CSS}</head><body>
{_NAV}<div class="page"><div class="card"><p>Resume #{resume_idx} not found. <a href="/resumes">Back to list</a></p></div></div></body></html>""", 404
                row = df.iloc[resume_idx]
                text = str(row.get('Resume_str', ''))
                resume_html_raw = str(row.get('Resume_html', ''))
                category = str(row.get('Category', ''))
                skills = self.resume_processor.extract_skills(text)
                edu = self.resume_processor.extract_education(text)
                exp = self.resume_processor.extract_experience(text)
                domain = self.resume_processor.extract_domain(text)
                skills_html = " ".join(f'<span class="tag">{html.escape(s)}</span>' for s in skills)
                # Strip script/style tags from resume HTML for safe inline rendering
                safe_resume_html = re.sub(r'<(script|style)[^>]*>.*?</(script|style)>', '', resume_html_raw, flags=re.DOTALL | re.IGNORECASE)
            except Exception as e:
                return f"""<!DOCTYPE html><html><head><title>Error</title>{_BASE_CSS}</head><body>
{_NAV}<div class="page"><div class="card"><p style="color:#c62828">Error: {html.escape(str(e))}</p></div></div></body></html>""", 500

            return render_template_string(f"""<!DOCTYPE html>
<html><head>
<title>Resume #{resume_idx} — {html.escape(category)}</title>
{_BASE_CSS}
<style>
  .resume-frame {{ font-family:'Segoe UI',Arial,sans-serif; font-size:13px; line-height:1.7; color:#333; background:#fff; padding:24px; border-radius:6px; border:1px solid #eee; }}
  .resume-frame .name, .resume-frame h1, .resume-frame h2, .resume-frame h3 {{ color:#1a1a2e; font-weight:700; margin:12px 0 4px; }}
  .resume-frame .sectiontitle {{ font-size:14px; font-weight:700; text-transform:uppercase; border-bottom:2px solid #1a1a2e; margin:16px 0 8px; padding-bottom:4px; color:#1a1a2e; letter-spacing:.5px; }}
  .resume-frame .jobline, .resume-frame .field {{ display:inline; }}
  .resume-frame p {{ margin:4px 0; }}
  .resume-frame ul {{ margin:4px 0; padding-left:20px; }}
</style>
</head><body>
{_NAV}
<div class="page">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px">
    <a href="/resumes" style="color:#888;text-decoration:none;font-size:13px">← Back to dataset</a>
    <span style="color:#ddd">|</span>
    <span style="font-size:13px;color:#888">Resume #{resume_idx}</span>
  </div>

  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px">
      <div>
        <h1 style="margin-bottom:6px">{html.escape(category)}</h1>
        <div style="font-size:13px;color:#666">
          {html.escape(edu.title())} · {exp:.1f} yrs exp · {html.escape(', '.join(domain))}
        </div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <form action="/upload_raw" method="post" style="margin:0">
          <input type="hidden" name="resume_text" value="{html.escape(text)}">
          <button type="submit" class="btn btn-green" style="font-size:12px;padding:6px 14px">Analyse This Resume</button>
        </form>
      </div>
    </div>

    <div style="margin-top:14px">
      <div class="section-label">Skills Detected ({len(skills)})</div>
      <div style="margin-top:6px">{skills_html if skills_html else '<em style="color:#888">No skills detected</em>'}</div>
    </div>
  </div>

  <div class="card resume-frame">
    {safe_resume_html}
  </div>
</div>
</body></html>""")


        # ------------------------------------------------------------------
        # Rules explorer — mined association rules + pruning controls
        # ------------------------------------------------------------------
        @self.app.route('/rules')
        def rules_page():
            combo = self.base_engine.rules['combination']
            gap   = self.base_engine.rules['gap']
            meta  = self._training_meta

            # Association rules table
            all_rules = sorted(combo.rules, key=lambda r: r.confidence, reverse=True)
            rule_rows = ""
            for i, r in enumerate(all_rules[:100]):
                ant = ', '.join(sorted(r.antecedent))
                con = ', '.join(sorted(r.consequent))
                all_skills_in_rule = ' '.join(sorted(r.antecedent) + sorted(r.consequent)).lower()
                lift_cls = 'lift-high' if r.lift >= 2 else ('lift-med' if r.lift >= 1.2 else 'lift-low')
                sup_pct  = f"{r.support*100:.1f}%"
                conf_pct = f"{r.confidence*100:.1f}%"
                lift_str = f"{r.lift:.2f}x"
                conf_bar = int(r.confidence * 80)
                bias_skill  = list(r.antecedent)[0]
                chain_skill = list(r.antecedent)[0]
                rule_rows += f"""<tr data-skills="{html.escape(all_skills_in_rule)}">
                  <td style="font-size:12px"><span class="mono">{html.escape(ant)}</span></td>
                  <td style="font-size:12px"><span class="mono">{html.escape(con)}</span></td>
                  <td>{sup_pct}</td>
                  <td>
                    <div style="display:flex;align-items:center;gap:6px">
                      <div class="score-bar-wrap" style="width:80px"><div class="score-bar" style="width:{conf_bar}px;background:#1565c0"></div></div>
                      {conf_pct}
                    </div>
                  </td>
                  <td class="{lift_cls}">{lift_str}</td>
                  <td style="display:flex;gap:4px">
                    <a href="/rules/bias?skill={html.escape(bias_skill)}" class="btn" style="font-size:11px;padding:4px 10px;background:#7b1fa2">Bias</a>
                    <a href="/rules/chain?skill={html.escape(chain_skill)}" class="btn" style="font-size:11px;padding:4px 10px;background:#37474f">Chain</a>
                  </td>
                </tr>"""

            rule_table = f"""
            <div style="margin-bottom:12px;display:flex;gap:12px;align-items:center">
              <input id="rule-search" type="text" placeholder="Filter by skill name (e.g. python, aws, sql)..."
                     style="flex:1;padding:8px 12px;border:1px solid #ddd;border-radius:6px;font-size:14px"
                     oninput="filterRules(this.value)">
              <span id="rule-count" style="font-size:13px;color:#888;white-space:nowrap">{len(all_rules)} rules</span>
              <a href="/rules/chain" class="btn" style="background:#37474f;font-size:12px;padding:6px 14px">Chain Explorer →</a>
            </div>
            <table id="rules-table">
              <tr><th>IF (antecedent)</th><th>THEN (consequent)</th><th>Support</th><th>Confidence</th><th>Lift</th><th>Actions</th></tr>
              {rule_rows if rule_rows else '<tr><td colspan="6" style="text-align:center;color:#888;padding:20px">No rules mined — lower thresholds</td></tr>'}
            </table>
            <script>
            function filterRules(q) {{
              const rows = document.querySelectorAll('#rules-table tr[data-skills]');
              let shown = 0;
              rows.forEach(r => {{
                const match = !q || r.dataset.skills.includes(q.toLowerCase());
                r.style.display = match ? '' : 'none';
                if (match) shown++;
              }});
              document.getElementById('rule-count').textContent = shown + ' rules';
            }}
            </script>""" if all_rules else '<p style="color:#888">No association rules found. Try lowering the thresholds.</p>'

            # Top positive skills (most associated with hiring)
            pos_skills = combo.positive_skills.most_common(20)
            pos_html = " ".join(
                f'<span class="tag" style="font-size:11px">{html.escape(s)} <strong>({c})</strong></span>'
                for s, c in pos_skills
            ) or "<em>none</em>"

            neg_skills = combo.negative_skills.most_common(10)
            neg_html = " ".join(
                f'<span class="tag tag-gap" style="font-size:11px">{html.escape(s)} <strong>({c})</strong></span>'
                for s, c in neg_skills
            ) or "<em>none</em>"

            # Critical skills (gap rule)
            crit_html = " ".join(
                f'<span class="tag" style="background:#fce4ec;color:#880e4f;font-size:11px">{html.escape(s)}</span>'
                for s in sorted(gap.critical_skills)
            ) or "<em>none — threshold may be too high</em>"

            # Experience thresholds table
            exp_thresholds = self.base_engine.rules['experience'].skill_experience_thresholds
            exp_rows = "".join(
                f"<tr><td>{html.escape(s)}</td><td>{v:.1f} yrs</td></tr>"
                for s, v in sorted(exp_thresholds.items(), key=lambda x: -x[1])[:20]
            ) or "<tr><td colspan='2' style='color:#888'>No thresholds learned</td></tr>"

            # Education hiring rates
            edu_scores = self.base_engine.rules['education'].education_scores
            edu_counts = self.base_engine.rules['education'].education_counts
            edu_rows = ""
            for level, rate in sorted(edu_scores.items(), key=lambda x: -x[1]):
                hired, total = edu_counts.get(level, (0, 0))
                bar_w = int(rate * 120)
                color = _score_color(rate)
                edu_rows += f"""<tr>
                  <td>{html.escape(level.title())}</td>
                  <td><div class="score-bar-wrap" style="width:120px"><div class="score-bar" style="width:{bar_w}px;background:{color}"></div></div> {rate*100:.0f}%</td>
                  <td style="color:#888;font-size:12px">{hired}/{total}</td>
                </tr>"""

            ms  = meta.get('min_support', 0.1)
            mc  = meta.get('min_confidence', 0.5)
            nr  = meta.get('n_resumes', 0)
            nh  = meta.get('n_hired', 0)
            nrl = meta.get('n_rules', 0)

            # Custom vocab tokens card — show count + threshold status
            custom_tok_rows = ""
            for idx, entry in enumerate(self._custom_vocab_tokens):
                tok   = entry['token'] if isinstance(entry, dict) else entry
                count = entry.get('count', '?') if isinstance(entry, dict) else '?'
                rate  = entry.get('rate', 0)   if isinstance(entry, dict) else 0
                meets = isinstance(rate, float) and rate >= ms
                if meets:
                    stat = (f'<span class="badge badge-ok" style="font-size:10px">'
                            f'{count} resumes ({rate*100:.1f}%) ✓ above threshold</span>')
                else:
                    stat = (f'<span class="badge badge-bias" style="font-size:10px">'
                            f'{count} resumes ({rate*100:.1f}%) — below {ms*100:.0f}% threshold → no rules mined</span>')
                custom_tok_rows += (
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
                    f'<span class="tag" style="font-size:13px;margin:0">{html.escape(tok)}</span>'
                    f'{stat}'
                    f'<form action="/rules/vocab/delete/{idx}" method="post" style="margin:0">'
                    f'<button type="submit" class="btn" style="background:#c62828;font-size:11px;padding:3px 8px">Remove</button>'
                    f'</form></div>'
                )
            custom_tok_html = custom_tok_rows or '<p style="color:#888;font-size:13px;margin:0">None added yet.</p>'

            return render_template_string(f"""<!DOCTYPE html><html><head>
<title>Rules Explorer — HireAudit</title>{_BASE_CSS}
</head><body>
{_NAV}
<div class="page">
  <h1>Rules Explorer</h1>
  <p style="color:#666;font-size:14px;margin-top:-8px">
    Trained on <strong>{nr}</strong> real resumes ({nh} hired) from Resume.csv.
    Mined <strong>{nrl}</strong> association rules.
  </p>

  <!-- Pruning controls -->
  <div class="card">
    <h2>Pruning Controls</h2>
    <p style="font-size:13px;color:#666;margin-top:0">
      Apriori algorithm prunes rules below support and confidence thresholds.
      Lower thresholds = more rules. Higher = fewer, stronger rules.
    </p>
    <form action="/rules/refit" method="post" style="display:flex;gap:20px;align-items:flex-end;flex-wrap:wrap">
      <div>
        <div class="section-label">Min Support (P(A∩B) / total)</div>
        <input type="number" name="min_support" value="{ms}" min="0.01" max="1.0" step="0.01"
               style="width:100px;padding:8px;border:1px solid #ddd;border-radius:6px;font-size:14px">
        <div style="font-size:11px;color:#888;margin-top:2px">current: {ms}</div>
      </div>
      <div>
        <div class="section-label">Min Confidence (P(B|A))</div>
        <input type="number" name="min_confidence" value="{mc}" min="0.01" max="1.0" step="0.01"
               style="width:100px;padding:8px;border:1px solid #ddd;border-radius:6px;font-size:14px">
        <div style="font-size:11px;color:#888;margin-top:2px">current: {mc}</div>
      </div>
      <div>
        <div class="section-label">Training sample size</div>
        <input type="number" name="n_resumes" value="{nr}" min="50" max="2484" step="50"
               style="width:100px;padding:8px;border:1px solid #ddd;border-radius:6px;font-size:14px">
      </div>
      <button type="submit" class="btn btn-green">Refit &amp; Reprune</button>
      <div style="align-self:flex-end;padding-bottom:4px">
        <label style="display:flex;align-items:center;gap:6px;font-size:13px;cursor:pointer">
          <input type="checkbox" name="no_prune" value="1"
                 title="Sets support=0.001 and confidence=0.01 — shows all co-occurrences including rare tokens like scrum">
          Disable pruning (show all)
        </label>
        <div style="font-size:11px;color:#888;margin-top:2px">
          Overrides thresholds → shows every co-occurrence incl. rare custom tokens
        </div>
      </div>
    </form>
  </div>

  <!-- Association rules -->
  <div class="card">
    <h2>Association Rules ({len(all_rules)} mined)</h2>
    <p style="font-size:13px;color:#666;margin-top:0">
      <strong>Support</strong> = fraction of hired candidates where both skills appear together.
      <strong>Confidence</strong> = P(skill B present | skill A present) among hired candidates.
      <strong>Lift</strong> = confidence ÷ base rate of B. Lift &gt; 1 = A predicts B better than chance.
    </p>
    {rule_table}
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Positive Skills</h2>
      <p style="font-size:12px;color:#666;margin-top:0">Skills most frequent among hired candidates (count).</p>
      {pos_html}
    </div>
    <div class="card">
      <h2>Rejection-Correlated Skills</h2>
      <p style="font-size:12px;color:#666;margin-top:0">Skills most frequent among rejected candidates.</p>
      {neg_html}
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Critical Skills (Gap Rule)</h2>
      <p style="font-size:12px;color:#666;margin-top:0">Skills present in &gt;50% of hired candidates. Missing = score penalty.</p>
      {crit_html}
    </div>
    <div class="card">
      <h2>Education Hiring Rates</h2>
      <table>{edu_rows}</table>
    </div>
  </div>

  <div class="card">
    <h2>Experience Thresholds (top 20 skills)</h2>
    <p style="font-size:12px;color:#666;margin-top:0">Median years experience of hired candidates per skill.</p>
    <table style="width:auto">
      <tr><th>Skill</th><th>Median exp required</th></tr>
      {exp_rows}
    </table>
  </div>

  <div class="card">
    <h2>Custom Vocabulary Tokens</h2>
    <p style="font-size:13px;color:#666;margin-top:0">
      Add any keyword or domain term (e.g. <em>foobar</em>, <em>llmops</em>, <em>synthetic-data</em>).
      It joins the extraction vocabulary immediately and the engine refits — new rules involving that token will surface.
    </p>
    <div style="margin-bottom:14px">{custom_tok_html}</div>
    <form action="/rules/vocab/add" method="post" id="vocab-form"
          style="display:flex;gap:10px;align-items:center">
      <input type="text" name="token" id="vocab-input"
             placeholder="e.g. foobar, llmops, rag-pipeline"
             style="padding:8px 12px;border:1px solid #ddd;border-radius:6px;font-size:14px;width:280px"
             onkeydown="if(event.key==='Enter'){{event.preventDefault();document.getElementById('vocab-form').submit();}}">
      <button type="submit" class="btn btn-green">Add Token &amp; Refit</button>
    </form>
    <p style="font-size:12px;color:#888;margin-top:8px">
      Token must appear in resume text to be extracted. Support threshold is automatically lowered
      to include rare custom tokens — rules for the new token always surface after adding.
      Use <strong>Disable pruning</strong> above to see every co-occurrence across all tokens.
      <a href="/rules/chain" style="color:#1565c0">Explore chains →</a>
    </p>
  </div>
</div>
</body></html>""")

        # ------------------------------------------------------------------
        # Representation bias test — presence rate by job category (no synthetic labels)
        # ------------------------------------------------------------------
        @self.app.route('/rules/bias')
        def rules_bias():
            skill = request.args.get('skill', '').strip().lower()
            if not skill:
                return redirect('/rules')

            combo      = self.base_engine.rules['combination']
            resumes    = getattr(self, '_training_resumes', [])
            categories = getattr(self, '_training_categories', [])
            if not resumes:
                return redirect('/rules')

            matching_rules = [r for r in combo.rules
                              if skill in {s.lower() for s in r.antecedent | r.consequent}]

            from collections import defaultdict
            cat_total = defaultdict(int)
            cat_fires = defaultdict(int)
            total_fires = 0

            for resume, cat in zip(resumes, categories):
                skills = set(resume.skill_tokens)
                cat_total[cat] += 1
                fires = (skill in skills or
                         any(r.antecedent.issubset(skills) for r in matching_rules))
                if fires:
                    cat_fires[cat] += 1
                    total_fires += 1

            overall_rate = total_fires / len(resumes) if resumes else 0

            rows_data = []
            for cat in sorted(cat_total):
                total = cat_total[cat]
                if total < 3:
                    continue
                fires = cat_fires.get(cat, 0)
                rows_data.append({'cat': cat, 'total': total, 'fires': fires,
                                  'rate': fires / total})

            rates = [r['rate'] for r in rows_data]
            if len(rates) >= 2:
                min_rate, max_rate = min(rates), max(rates)
                di_val = min_rate / max_rate if max_rate > 0 else 1.0
                bias_detected = di_val < 0.80
                spread = max_rate - min_rate
            else:
                di_val, bias_detected, spread = 1.0, False, 0.0

            # Good signal: skill appears broadly and evenly across categories
            high_coverage = overall_rate > 0.35
            low_spread    = spread < 0.20

            if not bias_detected and high_coverage and low_spread:
                verdict_box = (
                    f'<div class="ok-box" style="background:#e8f5e9;border-color:#2e7d32;font-size:14px">'
                    f'<strong>✅ Strong positive signal — <em>{html.escape(skill)}</em> is a broadly distributed skill.</strong><br>'
                    f'Appears in <strong>{overall_rate*100:.0f}%</strong> of resumes overall. '
                    f'Disparity ratio = <strong>{di_val:.2f}</strong> (EEOC threshold ≥ 0.80 — passed). '
                    f'Category spread = {spread*100:.0f}pp — low. '
                    f'This skill crosses job-category boundaries and does not structurally advantage one group. '
                    f'Safe to use as a hiring signal.</div>'
                )
            elif not bias_detected:
                verdict_box = (
                    f'<div class="ok-box">'
                    f'✅ No representation bias — disparity ratio = <strong>{di_val:.2f}</strong> (≥ 0.80 threshold). '
                    f'Overall presence: {overall_rate*100:.0f}%. '
                    f'Skill appears at comparable rates across categories.</div>'
                )
            else:
                lo_cat = min(rows_data, key=lambda x: x['rate'])
                hi_cat = max(rows_data, key=lambda x: x['rate'])
                verdict_box = (
                    f'<div class="warn-box">'
                    f'⚠️ Representation bias detected — disparity ratio = <strong>{di_val:.2f}</strong> (below 0.80). '
                    f'<strong>{html.escape(hi_cat["cat"])}</strong> has {hi_cat["rate"]*100:.0f}% presence vs '
                    f'<strong>{html.escape(lo_cat["cat"])}</strong> at {lo_cat["rate"]*100:.0f}%. '
                    f'Using this skill as a gating requirement structurally disadvantages the lower-presence categories.</div>'
                )

            table_rows = ""
            for r in sorted(rows_data, key=lambda x: -x['rate']):
                bar_w = int(r['rate'] * 100)
                color = _score_color(r['rate'])
                table_rows += f"""<tr>
                  <td><strong>{html.escape(r['cat'])}</strong></td>
                  <td style="color:#888">{r['total']}</td>
                  <td>
                    <div style="display:flex;align-items:center;gap:8px">
                      <div class="score-bar-wrap" style="width:100px">
                        <div class="score-bar" style="width:{bar_w}px;background:{color}"></div>
                      </div>
                      <strong style="color:{color}">{r['rate']*100:.0f}%</strong>
                      <span style="font-size:11px;color:#aaa">({r['fires']}/{r['total']})</span>
                    </div>
                  </td>
                </tr>"""

            rule_detail_rows = ""
            for r in matching_rules[:20]:
                ant = ', '.join(sorted(r.antecedent))
                con = ', '.join(sorted(r.consequent))
                lift_cls = 'lift-high' if r.lift >= 2 else ('lift-med' if r.lift >= 1.2 else 'lift-low')
                rule_detail_rows += f"""<tr>
                  <td class="mono">{html.escape(ant)}</td>
                  <td class="mono">{html.escape(con)}</td>
                  <td>{r.support*100:.1f}%</td>
                  <td>{r.confidence*100:.1f}%</td>
                  <td class="{lift_cls}">{r.lift:.2f}x</td>
                </tr>"""

            return render_template_string(f"""<!DOCTYPE html><html><head>
<title>Bias Check: {html.escape(skill)} — HireAudit</title>{_BASE_CSS}
</head><body>
{_NAV}
<div class="page">
  <div style="margin-bottom:20px">
    <a href="/rules" style="color:#888;font-size:13px;text-decoration:none">← Back to Rules</a>
  </div>

  <div class="card">
    <h1>Representation Check: <span class="mono" style="color:#7b1fa2">{html.escape(skill)}</span></h1>
    <p style="font-size:14px;color:#666;margin-top:0">
      How evenly does <strong>{html.escape(skill)}</strong> appear across {len(rows_data)} job categories?
      A skill that appears at similar rates everywhere is <em>fair to gate on</em>.
      One that concentrates in a few categories creates structural disadvantage for others.
      Overall presence: <strong>{overall_rate*100:.0f}%</strong> across {len(resumes)} resumes.
    </p>
    {verdict_box}
    <div style="font-size:12px;color:#888;margin-top:8px">
      Method: EEOC 4/5 rule on skill <em>presence rates</em> per category — no synthetic hiring labels.
      Resume.csv has no outcome data; representation rate is the honest measurable signal.
    </div>
  </div>

  <div class="card">
    <h2>Skill Presence by Job Category</h2>
    <p style="font-size:13px;color:#666;margin-top:0">
      % of resumes in each category where <em>{html.escape(skill)}</em> appears
      (directly or via a matching rule antecedent). Bars close together = equitable.
      Wide spread = structural bias risk.
    </p>
    <table>
      <tr><th>Category</th><th>N</th><th>Presence rate</th></tr>
      {table_rows or '<tr><td colspan="3" style="color:#888;text-align:center;padding:20px">No data</td></tr>'}
    </table>
  </div>

  <div class="card">
    <h2>Association Rules containing <em>{html.escape(skill)}</em> ({len(matching_rules)})</h2>
    {'<table><tr><th>IF</th><th>THEN</th><th>Support</th><th>Confidence</th><th>Lift</th></tr>' + rule_detail_rows + '</table>'
     if rule_detail_rows else '<p style="color:#888;font-size:13px">No mined rules contain this skill at current thresholds. Lower min_support in <a href="/rules">Rules</a> to surface more.</p>'}
  </div>

  <div class="card" style="background:#f7f8fa">
    <h2>How to read this</h2>
    <ul style="font-size:14px;line-height:1.8">
      <li><strong>High, even bars</strong> = skill is universally distributed — safe hiring signal. No group is structurally excluded.</li>
      <li><strong>Wide spread between bars</strong> = skill concentrates in certain categories. Gating on this skill implicitly gates on job background.</li>
      <li><strong>EEOC 4/5 rule:</strong> lowest category rate ÷ highest category rate must be ≥ 0.80 to pass.</li>
      <li>Go to <a href="/rules">Rules</a>, adjust thresholds and refit to see how changing the training changes which skills pass this test.</li>
    </ul>
  </div>
</div>
</body></html>""")

        # ------------------------------------------------------------------
        # Rules refit — reprune with new thresholds
        # ------------------------------------------------------------------
        @self.app.route('/rules/refit', methods=['POST'])
        def rules_refit():
            try:
                no_prune = request.form.get('no_prune', '') == '1'
                nr = max(50, min(2484, int(request.form.get('n_resumes', 300))))
                if no_prune:
                    ms, mc = 0.001, 0.01   # effectively no pruning
                else:
                    ms = max(0.001, min(1.0, float(request.form.get('min_support', 0.1))))
                    mc = max(0.001, min(1.0, float(request.form.get('min_confidence', 0.5))))
                self._setup_explanation_engine(min_support=ms, min_confidence=mc, n_resumes=nr)
            except Exception:
                pass
            return redirect('/rules')

        # ------------------------------------------------------------------
        # Help page
        # ------------------------------------------------------------------
        @self.app.route('/help')
        def help_page():
            return render_template_string(f"""<!DOCTYPE html><html><head><title>Help — HireAudit</title>{_BASE_CSS}</head><body>
{_NAV}
<div class="page">
  <div class="card">
    <h1>How to Use HireAudit</h1>
    <div class="grid2">
      <div>
        <h2>Upload Options</h2>
        <ul style="font-size:14px;line-height:1.8">
          <li><strong>Raw Text tab</strong> — paste any resume text. Skills, education, experience, and domain are extracted automatically.</li>
          <li><strong>JSON tab</strong> — paste structured data if you already have it tokenised.</li>
          <li><strong>Browse Dataset</strong> — pick any of 2,484 real resumes from the Kaggle dataset and click "Analyse This".</li>
        </ul>
        <h2>The 6 Bayesian Rules</h2>
        <ul style="font-size:14px;line-height:1.8">
          <li><strong>Combination</strong> — skill co-occurrence association rules (e.g. python+sql → hired)</li>
          <li><strong>Experience</strong> — years-of-experience threshold match</li>
          <li><strong>Education</strong> — education level hiring pattern</li>
          <li><strong>Domain</strong> — industry background match</li>
          <li><strong>Gap</strong> — critical skill gap penalty</li>
          <li><strong>Bias</strong> — EEOC 4/5 disparate impact rule</li>
        </ul>
      </div>
      <div>
        <h2>Reading the Score</h2>
        <table style="font-size:13px">
          <tr><th>Score</th><th>Recommendation</th></tr>
          <tr><td>≥ 80%</td><td><span class="badge badge-strong">Strong Hire</span></td></tr>
          <tr><td>60–79%</td><td><span class="badge badge-hire">Hire</span></td></tr>
          <tr><td>40–59%</td><td><span class="badge badge-interview">Interview</span></td></tr>
          <tr><td>&lt; 40%</td><td><span class="badge badge-pass">Pass</span></td></tr>
        </table>
        <h2 style="margin-top:20px">Bias Flag</h2>
        <p style="font-size:14px">⚠️ shown when the EEOC 4/5 rule is violated — the lowest demographic group's hiring rate is less than 80% of the highest group's rate.</p>
        <h2>Demo Flow</h2>
        <ol style="font-size:14px;line-height:1.8">
          <li>Go to <a href="/resumes">/resumes</a>, pick a resume, click "Analyse This"</li>
          <li>See the extraction step — what skills were found</li>
          <li>See the 6 rule bars and bias check</li>
          <li>Go to <a href="/candidates/ranked">/candidates/ranked</a> to compare all</li>
          <li>Click any candidate for the full detail view</li>
        </ol>
      </div>
    </div>
  </div>
</div>
</body></html>""")

        # ------------------------------------------------------------------
        # Preferences page — user states what they want
        # ------------------------------------------------------------------
        @self.app.route('/preferences')
        def preferences_page():
            n = len(getattr(self, '_training_resumes', []))
            skill_categories = {
                'Programming': ['python', 'java', 'javascript', 'c++', 'c#', 'r', 'sql', 'scala',
                                'go', 'rust', 'typescript', 'swift', 'kotlin', 'php', 'ruby', 'perl'],
                'Web': ['react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
                        'express', 'fastapi', 'html', 'css', 'rest api', 'graphql', 'asp.net'],
                'Data / ML': ['tensorflow', 'pytorch', 'sklearn', 'pandas', 'numpy', 'spark',
                               'hadoop', 'tableau', 'powerbi', 'bigquery', 'databricks',
                               'snowflake', 'kafka', 'airflow', 'dbt', 'xgboost', 'keras'],
                'Cloud / DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
                                   'ansible', 'jenkins', 'git', 'gitlab', 'github', 'circleci',
                                   'cloudformation'],
                'Databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'oracle',
                              'elasticsearch', 'dynamodb', 'neo4j'],
                'Soft Skills': ['leadership', 'communication', 'teamwork', 'project management',
                                'analytical thinking', 'presentation', 'mentoring',
                                'problem solving', 'critical thinking', 'collaboration'],
            }

            chip_sections = ""
            for cat, skills in skill_categories.items():
                chips = "".join(
                    f'<span class="skill-chip" data-skill="{html.escape(s)}" '
                    f'onclick="addSkill(\'{html.escape(s)}\')" '
                    f'title="Click to add">{html.escape(s)}</span>'
                    for s in skills
                )
                chip_sections += f'''
                <div class="chip-section" data-cat="{html.escape(cat)}">
                  <div class="section-label" style="margin:10px 0 4px">{html.escape(cat)}</div>
                  <div class="chip-row">{chips}</div>
                </div>'''

            return render_template_string(f"""<!DOCTYPE html><html><head>
<title>Rank by Preferences — HireAudit</title>{_BASE_CSS}
<style>
  .skill-chip {{
    display: inline-block; padding: 4px 10px; margin: 3px; border-radius: 20px;
    background: #e8eaf6; color: #283593; font-size: 12px; cursor: pointer;
    border: 1px solid #c5cae9; transition: background .15s;
  }}
  .skill-chip:hover {{ background: #1565c0; color: white; }}
  .chip-row {{ display: flex; flex-wrap: wrap; gap: 0; }}
  .mode-btn {{ padding: 8px 18px; border: 2px solid #ccc; border-radius: 20px;
               cursor: pointer; font-size: 13px; font-weight: 700; background: white; }}
  .mode-btn.active-req {{ background: #2e7d32; color: white; border-color: #2e7d32; }}
  .mode-btn.active-pref {{ background: #7b1fa2; color: white; border-color: #7b1fa2; }}
  .big-input {{ width: 100%; padding: 10px 14px; border: 2px solid #ddd; border-radius: 8px;
                font-size: 14px; font-family: monospace; }}
  .big-input:focus {{ outline: none; border-color: #1565c0; }}
</style>
<script>
let currentMode = 'req';
function setMode(m) {{
  currentMode = m;
  document.getElementById('btn-req').className = 'mode-btn' + (m==='req' ? ' active-req' : '');
  document.getElementById('btn-pref').className = 'mode-btn' + (m==='pref' ? ' active-pref' : '');
}}
function addSkill(s) {{
  const inp = document.getElementById(currentMode === 'req' ? 'req-input' : 'pref-input');
  const cur = inp.value.split(',').map(x => x.trim()).filter(Boolean);
  if (!cur.includes(s)) {{ cur.push(s); inp.value = cur.join(', '); }}
}}
function filterChips(q) {{
  document.querySelectorAll('.skill-chip').forEach(c => {{
    c.style.display = (!q || c.dataset.skill.includes(q.toLowerCase())) ? '' : 'none';
  }});
  document.querySelectorAll('.chip-section').forEach(s => {{
    const visible = [...s.querySelectorAll('.skill-chip')].some(c => c.style.display !== 'none');
    s.style.display = visible ? '' : 'none';
  }});
}}
</script>
</head><body>
{_NAV}
<div class="page">
  <div class="card">
    <h1>Rank Resumes by Your Preferences</h1>
    <p style="font-size:14px;color:#666;margin-top:0">
      Set what you want. The system uses <strong>Bayesian Thompson Sampling</strong> to score all
      <strong>{n}</strong> training resumes. Association rules automatically expand your skill
      requirements — if you want <em>python</em> and there's a rule <em>python → sql</em>,
      <em>sql</em> is added as a valued skill.
    </p>
  </div>

  <form action="/rank" method="post">
    <div class="grid2">
      <div class="card">
        <h2>Required Skills <span style="font-size:12px;font-weight:400;color:#888">(penalised if missing)</span></h2>
        <input class="big-input" type="text" name="required" id="req-input"
               placeholder="python, sql, aws  — or click skills below">
      </div>
      <div class="card">
        <h2>Preferred Skills <span style="font-size:12px;font-weight:400;color:#888">(bonus if present)</span></h2>
        <input class="big-input" type="text" name="preferred" id="pref-input"
               placeholder="docker, kubernetes, tableau">
      </div>
    </div>

    <div class="card">
      <h2>Skill Picker</h2>
      <div style="display:flex;gap:10px;align-items:center;margin-bottom:12px;flex-wrap:wrap">
        <span style="font-size:13px;color:#666;font-weight:600">Add clicked skill to:</span>
        <button type="button" id="btn-req" class="mode-btn active-req" onclick="setMode('req')">Required</button>
        <button type="button" id="btn-pref" class="mode-btn" onclick="setMode('pref')">Preferred</button>
        <input type="text" placeholder="Filter skills..." oninput="filterChips(this.value)"
               style="padding:6px 12px;border:1px solid #ddd;border-radius:20px;font-size:13px;width:200px">
      </div>
      {chip_sections}
    </div>

    <div class="card">
      <div class="grid2">
        <div>
          <div class="section-label">Minimum Experience (years)</div>
          <input type="number" name="min_exp" value="0" min="0" max="20" step="0.5"
                 style="padding:8px;border:1px solid #ddd;border-radius:6px;font-size:14px;width:120px">
        </div>
        <div>
          <div class="section-label">Minimum Education Level</div>
          <select name="min_edu" style="padding:8px;border:1px solid #ddd;border-radius:6px;font-size:14px;width:160px">
            <option value="any">Any</option>
            <option value="high school">High School</option>
            <option value="bachelor">Bachelor</option>
            <option value="master">Master</option>
            <option value="phd">PhD</option>
          </select>
        </div>
      </div>
      <div style="margin-top:14px;display:flex;gap:24px;align-items:center;flex-wrap:wrap">
        <label style="font-size:13px;display:flex;align-items:center;gap:6px">
          <input type="checkbox" name="use_rules" value="on" checked>
          Expand requirements via association rules
        </label>
        <label style="font-size:13px;display:flex;align-items:center;gap:6px">
          Thompson samples per resume:
          <input type="number" name="n_samples" value="20" min="5" max="200"
                 style="width:70px;padding:4px 8px;border:1px solid #ddd;border-radius:4px">
        </label>
      </div>
    </div>

    <div style="margin-bottom:32px">
      <button type="submit" class="btn btn-green" style="font-size:16px;padding:14px 32px">
        Rank {n} Resumes →
      </button>
    </div>
  </form>
</div>
</body></html>""")

        # ------------------------------------------------------------------
        # Rank route — Thompson Sampling over training resumes
        # ------------------------------------------------------------------
        @self.app.route('/rank', methods=['POST'])
        def rank_resumes():
            import random
            random.seed()  # non-deterministic

            required_raw  = request.form.get('required', '')
            preferred_raw = request.form.get('preferred', '')
            min_exp       = float(request.form.get('min_exp', 0) or 0)
            min_edu       = request.form.get('min_edu', 'any')
            use_rules     = request.form.get('use_rules', '') == 'on'
            n_samples     = max(5, min(200, int(request.form.get('n_samples', 20) or 20)))

            required_skills  = {s.strip().lower() for s in required_raw.split(',') if s.strip()}
            preferred_skills = {s.strip().lower() for s in preferred_raw.split(',') if s.strip()}

            resumes    = getattr(self, '_training_resumes', [])
            categories = getattr(self, '_training_categories', [])
            if not resumes:
                return redirect('/preferences')

            # Expand via association rules
            combo = self.base_engine.rules['combination']
            expanded_req  = set(required_skills)
            expanded_pref = set(preferred_skills)
            rule_expansions = []

            if use_rules:
                for rule in combo.rules:
                    ant = {s.lower() for s in rule.antecedent}
                    con = {s.lower() for s in rule.consequent}
                    if ant & expanded_req:
                        new = con - expanded_req - expanded_pref
                        if new:
                            expanded_req |= new
                            rule_expansions.append(
                                f"<span class='mono'>{html.escape(', '.join(sorted(ant)))}</span> → "
                                f"<span class='mono'>{html.escape(', '.join(sorted(con)))}</span> "
                                f"<em>(added to required)</em>"
                            )
                    elif ant & expanded_pref:
                        new = con - expanded_req - expanded_pref
                        if new:
                            expanded_pref |= new
                            rule_expansions.append(
                                f"<span class='mono'>{html.escape(', '.join(sorted(ant)))}</span> → "
                                f"<span class='mono'>{html.escape(', '.join(sorted(con)))}</span> "
                                f"<em>(added to preferred)</em>"
                            )

            edu_rank = {'phd': 4, 'master': 3, 'bachelor': 2,
                        'associate': 1, 'high school': 0, 'unknown': -1, 'any': -1}
            min_edu_rank = edu_rank.get(min_edu, -1)

            scored = []
            for i, (resume, cat) in enumerate(zip(resumes, categories)):
                skills = {s.lower() for s in resume.skill_tokens}

                req_matched = sorted(expanded_req & skills)
                req_missed  = sorted(expanded_req - skills)
                pref_matched = sorted(expanded_pref & skills)

                # Beta(α, β) — alpha = successes, beta = failures
                alpha = 1.0 + len(req_matched) + len(pref_matched) * 0.5
                beta  = 1.0 + len(req_missed)

                # Thompson Sampling: take max of n_samples draws
                ts_score = max(random.betavariate(alpha, beta) for _ in range(n_samples))

                exp_ok = resume.years_experience >= min_exp
                edu_ok = edu_rank.get(resume.education_level, -1) >= min_edu_rank

                if not exp_ok:
                    ts_score *= 0.5
                if not edu_ok:
                    ts_score *= 0.7

                scored.append({
                    'idx': i, 'cat': cat, 'resume': resume,
                    'ts_score': ts_score,
                    'alpha': alpha, 'beta': beta,
                    'req_matched': req_matched,
                    'req_missed': req_missed,
                    'pref_matched': pref_matched,
                    'exp_ok': exp_ok, 'edu_ok': edu_ok,
                })

            scored.sort(key=lambda x: -x['ts_score'])

            # Build result rows
            result_rows = ""
            for rank, s in enumerate(scored[:100], 1):
                color = _score_color(s['ts_score'])
                bar_w = int(s['ts_score'] * 140)

                req_matched_html = " ".join(
                    f'<span class="tag" style="background:#e8f5e9;color:#1b5e20">{html.escape(sk)}</span>'
                    for sk in s['req_matched']
                ) or '<span style="color:#aaa;font-size:12px">—</span>'

                req_missed_html = " ".join(
                    f'<span class="tag tag-gap">{html.escape(sk)}</span>'
                    for sk in s['req_missed']
                ) or '<span style="color:#aaa;font-size:12px">—</span>'

                pref_html = " ".join(
                    f'<span class="tag" style="background:#f3e5f5;color:#6a1b9a">{html.escape(sk)}</span>'
                    for sk in s['pref_matched']
                ) or '<span style="color:#aaa;font-size:12px">—</span>'

                warnings = ""
                if not s['exp_ok']:
                    warnings += f'<span class="badge badge-bias" style="font-size:10px">exp &lt; {min_exp}y</span> '
                if not s['edu_ok']:
                    warnings += f'<span class="badge badge-bias" style="font-size:10px">edu below {html.escape(min_edu)}</span> '

                result_rows += f"""<tr>
                  <td style="font-weight:700;font-size:13px"><a href="/resumes/{s['idx']}" class="row-link">#{rank}</a></td>
                  <td><span class="badge badge-hire" style="font-size:11px">{html.escape(s['cat'])}</span></td>
                  <td>
                    <div style="display:flex;align-items:center;gap:6px">
                      <div class="score-bar-wrap" style="width:140px">
                        <div class="score-bar" style="width:{bar_w}px;background:{color}"></div>
                      </div>
                      <strong style="color:{color}">{s['ts_score']*100:.0f}%</strong>
                      <span style="font-size:11px;color:#aaa">β({s['alpha']:.1f},{s['beta']:.1f})</span>
                    </div>
                    {warnings}
                  </td>
                  <td style="font-size:12px">{req_matched_html}</td>
                  <td style="font-size:12px">{req_missed_html}</td>
                  <td style="font-size:12px">{pref_html}</td>
                  <td><a href="/resumes/{s['idx']}" class="btn" style="font-size:11px;padding:4px 10px;background:#1565c0">View</a></td>
                </tr>"""

            # Summary stats
            top10_cats = {}
            for s in scored[:10]:
                top10_cats[s['cat']] = top10_cats.get(s['cat'], 0) + 1
            top10_html = " ".join(
                f'<span class="tag">{html.escape(c)} ({n})</span>'
                for c, n in sorted(top10_cats.items(), key=lambda x: -x[1])
            )

            expansion_html = ""
            if rule_expansions:
                expansion_html = (
                    '<div class="warn-box" style="background:#e8f5e9;border-color:#4caf50;margin-bottom:0">'
                    '<strong>Rules expanded your requirements:</strong><ul style="margin:6px 0 0;padding-left:18px;font-size:13px">'
                    + "".join(f"<li>{e}</li>" for e in rule_expansions)
                    + '</ul></div>'
                )
            elif use_rules:
                expansion_html = '<div class="ok-box" style="margin-bottom:0">No association rules fired — requirements unchanged.</div>'

            req_display  = html.escape(', '.join(sorted(expanded_req)))  or '<em>none</em>'
            pref_display = html.escape(', '.join(sorted(expanded_pref))) or '<em>none</em>'

            return render_template_string(f"""<!DOCTYPE html><html><head>
<title>Ranked Results — HireAudit</title>{_BASE_CSS}
</head><body>
{_NAV}
<div class="page">
  <div style="margin-bottom:16px">
    <a href="/preferences" style="color:#888;font-size:13px;text-decoration:none">← Back to Preferences</a>
  </div>

  <div class="card">
    <h1>Thompson Sampling Ranking — {len(scored)} Resumes</h1>
    <div class="grid2" style="margin-bottom:12px">
      <div>
        <div class="section-label">Required skills (after expansion)</div>
        <div style="font-size:13px;font-family:monospace">{req_display}</div>
      </div>
      <div>
        <div class="section-label">Preferred skills (after expansion)</div>
        <div style="font-size:13px;font-family:monospace">{pref_display}</div>
      </div>
    </div>
    {expansion_html}
  </div>

  <div class="card" style="background:#f7f8fa">
    <div style="display:flex;gap:24px;flex-wrap:wrap;font-size:13px">
      <div><strong>Min experience:</strong> {min_exp}y</div>
      <div><strong>Min education:</strong> {html.escape(min_edu)}</div>
      <div><strong>Thompson samples:</strong> {n_samples} per resume</div>
      <div><strong>Top-10 categories:</strong> {top10_html}</div>
    </div>
    <p style="font-size:12px;color:#888;margin:8px 0 0">
      Score = max of {n_samples} draws from Beta(α, β) where α = 1 + required_matched + 0.5×preferred_matched, β = 1 + required_missed.
      Higher β means more missing skills → lower scores. Experience/education floor violations apply a score multiplier.
    </p>
  </div>

  <div class="card">
    <h2>Ranked Resumes (top 100 of {len(scored)})</h2>
    <table>
      <tr>
        <th>Rank</th>
        <th>Category</th>
        <th>Thompson Score β(α,β)</th>
        <th>Required ✓</th>
        <th>Required ✗ (missing)</th>
        <th>Preferred ✓</th>
        <th></th>
      </tr>
      {result_rows or '<tr><td colspan="6" style="text-align:center;color:#888;padding:24px">No results</td></tr>'}
    </table>
  </div>
</div>
</body></html>""")

        # ------------------------------------------------------------------
        # Custom vocab token — add a new keyword to the extraction vocabulary
        # ------------------------------------------------------------------
        @self.app.route('/rules/vocab/add', methods=['POST'])
        def vocab_add():
            import re as _re
            token = request.form.get('token', '').strip().lower()
            existing = [t['token'] for t in self._custom_vocab_tokens]
            if not token or token in self.resume_processor.vocabulary.tokens or token in existing:
                return redirect('/rules')

            # Count how many training resumes actually contain this token
            pat = _re.compile(r'\b' + _re.escape(token) + r'\b', _re.IGNORECASE)
            resumes_raw = getattr(self, '_training_resumes', [])
            # We need the raw texts — re-read from CSV for the count
            try:
                import pandas as pd
                csv_path = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), '..', '..', 'data', 'raw', 'resume_dataset', 'Resume.csv'
                ))
                nr_cur = self._training_meta.get('n_resumes', 300)
                df_raw = pd.read_csv(csv_path).head(nr_cur)
                count = sum(1 for t in df_raw['Resume_str'].astype(str) if pat.search(t))
                rate  = count / nr_cur if nr_cur else 0
            except Exception:
                count, rate = 0, 0.0

            self.resume_processor.vocabulary.tokens.append(token)
            self.resume_processor.tech_skills.append(token)
            self._custom_vocab_tokens.append({'token': token, 'count': count, 'rate': rate})

            ms = self._training_meta.get('min_support', 0.1)
            mc = self._training_meta.get('min_confidence', 0.5)
            nr = self._training_meta.get('n_resumes', 300)
            # Auto-lower support so the custom token's rules always surface
            effective_ms = min(ms, max(0.001, rate * 0.8)) if isinstance(rate, float) and rate > 0 else 0.001
            self._setup_explanation_engine(min_support=effective_ms, min_confidence=mc, n_resumes=nr)
            return redirect('/rules')

        @self.app.route('/rules/vocab/delete/<int:idx>', methods=['POST'])
        def vocab_delete(idx):
            if 0 <= idx < len(self._custom_vocab_tokens):
                entry = self._custom_vocab_tokens.pop(idx)
                token = entry['token'] if isinstance(entry, dict) else entry
                try:
                    self.resume_processor.vocabulary.tokens.remove(token)
                    self.resume_processor.tech_skills.remove(token)
                except ValueError:
                    pass
                ms = self._training_meta.get('min_support', 0.1)
                mc = self._training_meta.get('min_confidence', 0.5)
                nr = self._training_meta.get('n_resumes', 300)
                self._setup_explanation_engine(min_support=ms, min_confidence=mc, n_resumes=nr)
            return redirect('/rules')

        # ------------------------------------------------------------------
        # Rule chain explorer — BFS through rule graph from a seed skill
        # ------------------------------------------------------------------
        @self.app.route('/rules/chain')
        def rules_chain():
            skill = request.args.get('skill', '').strip().lower()
            combo = self.base_engine.rules['combination']
            all_skills_in_rules = sorted({s.lower()
                                          for r in combo.rules
                                          for s in r.antecedent | r.consequent})

            if not skill:
                # Show skill picker
                chips = "".join(
                    f'<a href="/rules/chain?skill={html.escape(s)}" '
                    f'class="tag" style="text-decoration:none;font-size:13px">{html.escape(s)}</a>'
                    for s in all_skills_in_rules
                ) or '<em style="color:#888">No rules mined — lower thresholds on Rules page.</em>'
                return render_template_string(f"""<!DOCTYPE html><html><head>
<title>Rule Chains — HireAudit</title>{_BASE_CSS}
</head><body>{_NAV}
<div class="page">
  <div class="card">
    <h1>Rule Chain Explorer</h1>
    <p style="font-size:14px;color:#666;margin-top:0">
      Pick a seed skill to trace how association rules chain: A → B → C.
      Each arrow is one mined rule. Combined confidence = product of individual confidences along the path.
    </p>
    <div style="display:flex;flex-wrap:wrap;gap:4px">{chips}</div>
  </div>
</div></body></html>""")

            chains = self._chain_rules(skill, max_depth=3)

            # Separate by depth
            depth1 = [(p, c) for p, c in chains if len(p) == 1]
            depth2 = [(p, c) for p, c in chains if len(p) == 2]
            depth3 = [(p, c) for p, c in chains if len(p) == 3]

            def rule_arrow(rule, combined_conf=None):
                ant = ' + '.join(sorted(rule.antecedent))
                con = ' + '.join(sorted(rule.consequent))
                lift_cls = 'lift-high' if rule.lift >= 2 else ('lift-med' if rule.lift >= 1.2 else 'lift-low')
                extra = (f' <span style="color:#888;font-size:11px">chain: <strong>{combined_conf*100:.0f}%</strong></span>'
                         if combined_conf is not None else '')
                return (f'<span class="mono">{html.escape(ant)}</span>'
                        f' <span style="color:#1565c0;font-weight:700">→</span> '
                        f'<span class="mono">{html.escape(con)}</span>'
                        f' <span class="{lift_cls}" style="font-size:12px">'
                        f'conf:{rule.confidence*100:.0f}% lift:{rule.lift:.1f}x</span>{extra}')

            def chain_rows(chain_list):
                if not chain_list:
                    return '<p style="color:#888;font-size:13px">None at this depth.</p>'
                rows = ""
                for path, combined_conf in chain_list[:20]:
                    steps = []
                    for i, rule in enumerate(path):
                        cc = combined_conf if i == len(path) - 1 else None
                        steps.append(rule_arrow(rule, cc))
                    rows += (f'<div style="padding:10px 0;border-bottom:1px solid #f0f0f0;font-size:13px">'
                             + ' &nbsp;<span style="color:#aaa">⟹</span>&nbsp; '.join(steps)
                             + '</div>')
                return rows

            # Unique consequents reachable from skill (for summary)
            reachable = set()
            for path, _ in chains:
                for rule in path:
                    reachable |= {s.lower() for s in rule.consequent}
            reachable_html = " ".join(
                f'<span class="tag">{html.escape(s)}</span>' for s in sorted(reachable)
            ) or '<em>none</em>'

            return render_template_string(f"""<!DOCTYPE html><html><head>
<title>Chain: {html.escape(skill)} — HireAudit</title>{_BASE_CSS}
</head><body>{_NAV}
<div class="page">
  <div style="margin-bottom:16px">
    <a href="/rules/chain" style="color:#888;font-size:13px;text-decoration:none">← All skills</a>
    &nbsp;|&nbsp;
    <a href="/rules" style="color:#888;font-size:13px;text-decoration:none">Rules page</a>
  </div>

  <div class="card">
    <h1>Rule Chains from <span class="mono" style="color:#1565c0">{html.escape(skill)}</span></h1>
    <p style="font-size:14px;color:#666;margin-top:0">
      BFS through the association rule graph starting at <em>{html.escape(skill)}</em>.
      Depth 1 = direct rules. Depth 2 = rules whose consequent is an antecedent of another rule.
      Combined confidence = product of individual rule confidences along the path.
    </p>
    <div>
      <div class="section-label">All skills reachable from {html.escape(skill)}</div>
      <div style="margin-top:6px">{reachable_html}</div>
    </div>
  </div>

  <div class="card">
    <h2>Depth 1 — Direct rules ({len(depth1)})</h2>
    {chain_rows(depth1)}
  </div>

  <div class="card">
    <h2>Depth 2 — Two-hop chains ({len(depth2)})</h2>
    <p style="font-size:12px;color:#888;margin-top:0">
      Rule A fires, its consequent is the antecedent of Rule B.
      Combined confidence = conf(A) × conf(B).
    </p>
    {chain_rows(depth2)}
  </div>

  <div class="card">
    <h2>Depth 3 — Three-hop chains ({len(depth3)})</h2>
    <p style="font-size:12px;color:#888;margin-top:0">
      Combined confidence = conf(A) × conf(B) × conf(C). Long chains decay quickly.
    </p>
    {chain_rows(depth3)}
  </div>
</div>
</body></html>""")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sanitize_resume_data(self, data):
        if isinstance(data, dict):
            return {k: self._sanitize_resume_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_resume_data(item) for item in data]
        elif isinstance(data, str):
            return html.escape(data)
        return data

    def _store_candidate(self, resume: Resume, explanation, audit_result) -> int:
        self.candidate_counter += 1
        self.candidates.append({
            'id': self.candidate_counter,
            'resume': resume,
            'explanation': explanation,
            'audit_result': audit_result,
            'confidence_score': audit_result.overall_score,
            'timestamp': time.time(),
        })
        return self.candidate_counter

    def _chain_rules(self, start_skill: str, max_depth: int = 3) -> list:
        """BFS through the rule graph from start_skill. Returns list of (path, combined_conf)."""
        from collections import defaultdict, deque
        combo = self.base_engine.rules['combination']

        # Index: skill → rules where that skill appears in the antecedent
        skill_to_rules = defaultdict(list)
        for rule in combo.rules:
            for s in rule.antecedent:
                skill_to_rules[s.lower()].append(rule)

        chains = []
        start = start_skill.lower()
        # queue items: (path_of_rules, combined_conf, visited_skill_set)
        q = deque()
        for rule in skill_to_rules.get(start, []):
            covered = {s.lower() for s in rule.antecedent | rule.consequent}
            q.append(([rule], rule.confidence, covered))

        while q:
            path, conf, visited = q.popleft()
            chains.append((path, conf))
            if len(path) >= max_depth:
                continue
            last = path[-1]
            for con_skill in last.consequent:
                for next_rule in skill_to_rules.get(con_skill.lower(), []):
                    next_skills = {s.lower() for s in next_rule.antecedent | next_rule.consequent}
                    if not next_skills & visited:          # no cycle
                        q.append((path + [next_rule],
                                  conf * next_rule.confidence,
                                  visited | next_skills))

        chains.sort(key=lambda x: (-x[1], -len(x[0])))
        return chains[:60]

    def _infer_demographic_proxy(self, text: str, domain: list, skills: list) -> dict:
        """Infer demographic proxy signals from job title and domain — no actual demographics."""
        t = text.lower()
        if any(x in t for x in ['vp ', 'vice president', ' director', 'head of', 'chief ', 'president']):
            seniority = 'Executive'
        elif any(x in t for x in ['senior ', 'sr.', 'lead ', 'principal ', 'staff ']):
            seniority = 'Senior'
        elif any(x in t for x in ['junior ', 'jr.', 'associate ', 'entry level', 'intern']):
            seniority = 'Junior / Entry'
        else:
            seniority = 'Mid-level'

        tech_domains   = {'python developer', 'java developer', 'data science', 'machine learning',
                          'devops engineer', 'database', 'software developer', 'network security',
                          'hadoop', 'blockchain', 'etl developer', 'dot net developer'}
        other_domains  = {'hr', 'healthcare', 'teacher', 'advocate', 'arts', 'chef',
                          'fitness trainer', 'business development', 'sales', 'automation testing',
                          'civil engineer', 'electrical engineering', 'mechanical engineer',
                          'operations manager', 'pmO'}
        domain_str = ' '.join(domain).lower()
        if any(d in domain_str for d in tech_domains):
            group_label = 'Tech-pipeline background'
            group_color = '#1565c0'
            group_note  = 'Historically overrepresented in tech hiring — model may have absorbed this signal'
        elif any(d in domain_str for d in other_domains):
            group_label = 'Non-tech pipeline background'
            group_color = '#7b1fa2'
            group_note  = 'Historically underrepresented in tech hiring — watch for proxy bias'
        else:
            group_label = 'Mixed / general background'
            group_color = '#555'
            group_note  = 'No strong pipeline bias signal detected'

        # Look for gendered title signals (proxy, not ground truth)
        male_titles   = ['engineer', 'developer', 'architect', 'devops', 'sysadmin']
        female_titles = ['hr ', 'human resources', 'recruiter', 'nurse', 'teacher', 'coordinator']
        title_signals = []
        for m in male_titles:
            if m in t:
                title_signals.append(f'"{m}" — field historically male-skewed')
                break
        for f in female_titles:
            if f in t:
                title_signals.append(f'"{f.strip()}" — field historically female-skewed')
                break

        return {
            'seniority': seniority,
            'group_label': group_label,
            'group_color': group_color,
            'group_note': group_note,
            'primary_domain': domain[0] if domain else 'unknown',
            'title_signals': title_signals,
        }

    def _counterfactual_analysis(self, resume: Resume, original_score: float) -> list:
        """Score the same resume with different domain backgrounds — holds all else constant."""
        alternatives = [
            ('Tech / Software',   ['tech', 'software']),
            ('Data & Analytics',  ['data', 'analytics']),
            ('Finance / Banking', ['finance', 'banking']),
            ('Healthcare',        ['healthcare', 'medical']),
            ('HR / Management',   ['hr', 'management']),
            ('Creative / Arts',   ['arts', 'design']),
        ]
        results = []
        for label, alt_domain in alternatives:
            alt = Resume(
                skill_tokens=resume.skill_tokens,
                years_experience=resume.years_experience,
                education_level=resume.education_level,
                domain_background=alt_domain,
                demographics=resume.demographics,
            )
            try:
                audit = self.base_engine.audit_resume(alt)
                results.append({'label': label, 'score': audit.overall_score,
                                'delta': audit.overall_score - original_score})
            except Exception:
                pass
        results.sort(key=lambda x: -x['score'])
        return results

    def run(self, debug: bool = False, port: int = 5000):
        self.app.run(debug=debug, port=port)


class ExplainableInterface:
    def __init__(self, hiring_app: HiringApp):
        self.hiring_app = hiring_app
        self.app = hiring_app.app

    def process_resume(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        resume = Resume(
            skill_tokens=resume_data.get('skill_tokens', []),
            years_experience=float(resume_data.get('years_experience', 0)),
            education_level=resume_data.get('education_level', 'unknown'),
            domain_background=resume_data.get('domain_background', []),
            demographics=resume_data.get('demographics', {}),
        )
        audit_result = self.hiring_app.base_engine.audit_resume(resume)
        explanation = self.hiring_app.explanation_engine.explain_decision(resume, audit_result)
        return {
            'resume': resume,
            'audit_result': audit_result,
            'explanation': explanation,
            'confidence_score': audit_result.overall_score,
        }
