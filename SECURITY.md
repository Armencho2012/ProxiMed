# Security Policy

We take security and privacy seriously in our medical AI applications. This document outlines how to report vulnerabilities and our support policy.

## Supported Versions

Only the latest major version is actively supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | :white_check_mark: |
| 1.x.x   | :x:                |

## Reporting a Vulnerability

If you identify a security vulnerability or privacy concern (such as patient data handling issues or algorithmic backdoors), please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please report vulnerabilities by:
- Creating a [GitHub Private Vulnerability Report](https://docs.github.com/en/code-security/security-advisories/guidelines-for-reporting-security-vulnerabilities/reporting-a-vulnerability-to-a-repository-maintainer) (if available).
- Alternatively, contacting the repository owner via GitHub profile contact options.

### What to Include
Please provide the following details:
- A clear description of the vulnerability or risk.
- Steps to reproduce the issue (and a proof-of-concept if available).
- Potential impact on medical screening outcomes or data privacy.

## Response Timeline

We commit to the following turnaround times for confirmed security issues:
- **Acknowledgement**: Within 48 hours of receipt.
- **Triage & Fix**: Within 7 business days for high-severity issues.

## Scope of Security Concerns
- **Data Privacy**: Leakage of self-reported diagnostic vectors or training logs.
- **Model Poisoning**: Manipulation of the offline CDC training dataset or input validation bypasses.
- **Adversarial Inputs**: Intentionally crafted input values that crash the GUI or bypass spatial calculations.

## Out of Scope
- Concerns regarding model prediction accuracy or clinical efficacy. Please refer to [DISCLAIMER.md](DISCLAIMER.md) for limitations on clinical use.
- General software bugs that do not pose security or privacy risks (please report these via regular Bug Reports).
