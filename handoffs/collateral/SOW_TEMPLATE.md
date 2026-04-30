# Statement of Work: Explainable Hiring Compliance PoC

**Client:** [BANK NAME]  
**Provider:** [PROVIDER NAME]  
**SOW Date:** [DATE]  
**SOW Number:** [SOW-XXXX-2024]

## Engagement Overview

This Statement of Work establishes a 12-week Proof of Concept (PoC) engagement to develop and validate an explainable hiring compliance system meeting Federal Reserve SR 11-7 model risk management requirements, NYC Local Law 144 bias audit mandates, and FCRA adverse action notice standards.

The system applies Bayesian rule reliability estimation to content-neutral resume features, creating per-skill aptitude scores with quantified uncertainty while maintaining full explainability and fairness monitoring capabilities.

## Scope

### In Scope: PoC Deliverables

1. **Automated Model Cards** compliant with SR 11-7 documentation standards
   - Model purpose, methodology, assumptions, and limitations
   - Performance metrics with confidence intervals
   - Fairness assessment and monitoring plan
   - Tier classification and validator sign-off sections

2. **Real-time Fairness Monitoring** dashboard with regulatory compliance
   - NYC LL144 4/5 rule violation alerts
   - Per-protected-class statistical tracking  
   - Intersectional analysis capabilities
   - Bootstrap confidence interval calculations

3. **FCRA-compliant Adverse Action Notices** auto-generated from rule contributions
   - Specific reason codes derived from firing rules
   - Required ECOA notice text
   - 60-day dispute window and consumer rights
   - Consumer Reporting Agency integration blocks

4. **Quarterly Bias Audit Reports** ready for regulatory submission
   - Disparate impact analysis with 4/5 rule annotation
   - Equalized odds gap measurement
   - Calibration and per-group AUC analysis
   - NYC LL144 machine-readable JSON appendix

5. **Per-skill Aptitude Explanations** for candidate feedback and manager guidance
   - Rule-based decomposition of hiring decisions
   - Quantified uncertainty intervals
   - Content-neutral feature explanations
   - Audit trail for all scoring components

## Out of Scope

- **Production deployment** beyond PoC validation environment
- **Integration with existing ATS/HRIS systems** beyond data pipeline demonstration  
- **Legal compliance certification** — independent legal review required for all outputs
- **Multi-role optimization** beyond 2-3 demonstration role types
- **Historical bias remediation analysis** of past hiring decisions
- **Performance guarantees above AUC 0.62** — methodology validation only
- **Multi-tenant SaaS deployment** — candidate data remains in client environment

## Timeline

### Phase 1: Foundation (Weeks 1-3)
**Deliverables:**
- Data pipeline setup and integration
- Content-neutral feature extraction implementation
- Baseline fairness metrics establishment
- Security and deployment environment configuration

**Acceptance Criteria:**
- Data pipeline processes 100+ candidate resumes
- Feature extractor produces content-neutral attributes
- Baseline bias metrics calculated per protected class

### Phase 2: Core Engine (Weeks 4-6)  
**Deliverables:**
- Association rule mining with fairness filtering
- Bayesian posterior reliability estimation
- Per-skill aptitude scoring system
- Rule firing explanation framework

**Acceptance Criteria:**
- Rules fire on ≥80% of candidate decisions
- Fairness filter removes protected attribute proxies
- Aptitude scores include quantified uncertainty

### Phase 3: User Experience (Weeks 7-9)
**Deliverables:**
- Candidate assessment interface with explanations
- Governance dashboard with real-time monitoring
- Counterfactual analysis capability
- Audit ledger implementation

**Acceptance Criteria:**
- Interface displays per-skill aptitude breakdowns
- Dashboard shows live fairness metrics
- All decisions logged with full audit trail

### Phase 4: Compliance Package (Weeks 10-12)
**Deliverables:**
- Automated model card generation
- Quarterly bias audit report system
- FCRA adverse action notice automation
- Production readiness assessment

**Acceptance Criteria:**
- Model cards meet SR 11-7 standards
- Bias reports satisfy NYC LL144 requirements  
- FCRA notices contain specific reason codes

## Pricing

**PoC Engagement Fee:** [TIER] — to be filled by client
- **Tier A:** $[XXX],000 — Full 12-week engagement
- **Tier B:** $[XXX],000 — Abbreviated 8-week validation
- **Tier C:** $[XXX],000 — Technical assessment only (4 weeks)

**Payment Schedule:** 25% at contract execution, 25% at Phase 2 completion, 25% at Phase 3 completion, 25% at final delivery

**Additional Costs:** Client responsible for infrastructure costs (AWS/Azure/GCP compute, storage, compliance tools)

## Assumptions

### Client Data Access
- Anonymized historical hiring data (≥1,000 decisions)
- Demographic field availability for fairness analysis
- Resume content in structured or semi-structured format
- Hiring outcome data with decision rationale

### Client Resources
- Dedicated MRM Model Validator for review cycles
- Legal Counsel availability for FCRA/ECOA validation
- HRIS Data Administrator for pipeline integration
- Security/TPRM Representative for deployment review

### MRM Cooperation Cadence  
- Weekly progress review with MRM stakeholder
- Bi-weekly model validation checkpoint meetings
- Real-time access to model documentation requirements
- Expedited review cycle for compliance deliverables

### Technical Environment
- Client VPC or on-premises deployment environment
- SSO/SAML authentication integration capability
- SIEM-compatible audit logging infrastructure
- KMS-managed secrets handling for sensitive data

## Termination Clauses

### Mutual Termination
Either party may terminate this engagement with 30-day written notice. Upon termination:
- Client pays for work completed through termination date
- Provider delivers all completed deliverables and work product
- Both parties return or destroy confidential information

### Termination for Cause
Either party may terminate immediately for material breach that remains uncured after 15-day notice period.

### Effect of Termination
- All intellectual property in methodology and algorithms remains with Provider
- All client data and customizations remain with Client  
- No ongoing support obligations beyond agreed delivery date

## Intellectual Property and Data Handling

### Data Handling
- **Client Data Residency:** All candidate and hiring data remains in client's environment
- **Data Processing:** Provider processes data within client's designated security boundary
- **Data Retention:** Provider retains no client data beyond engagement completion
- **Anonymization:** All demo materials use synthetic or properly anonymized data

### Intellectual Property
- **Methodology Ownership:** Provider retains all rights to core methodology and algorithms
- **Customizations:** Client owns all custom configurations and rule sets developed during engagement
- **Documentation:** Client owns all model cards, audit reports, and compliance deliverables
- **Source Code:** Provided under limited license for client's internal use only

### Security and Compliance
- **Encryption:** All data in transit and at rest encrypted to banking standards
- **Access Controls:** Role-based access with audit logging
- **Incident Response:** 4-hour acknowledgment, 24-hour incident report
- **Compliance Standards:** SOC 2 Type II and ISO 27001 framework adherence

## Signatures

**Client Representative:**  
Name: [PRINT NAME]  
Title: [TITLE]  
Signature: _________________________ Date: _________

**Provider Representative:**  
Name: [PRINT NAME]  
Title: [TITLE]  
Signature: _________________________ Date: _________

**Legal Review:**  
Approved by Client Legal: _________________________ Date: _________  
Approved by Provider Legal: _________________________ Date: _________

---

**Attachment A:** Technical Architecture Diagram  
**Attachment B:** Sample Deliverable Templates  
**Attachment C:** Security and Compliance Checklist