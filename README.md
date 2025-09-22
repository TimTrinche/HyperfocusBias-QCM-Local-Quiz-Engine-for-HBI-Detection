HyperfocusBias-QCM: Local Quiz Engine for HBI Detection

This repository contains a **local multiple-choice questionnaire (MCQ) tool** developed for research on the **Hyperfocus Bias (HBI)**.  
It was designed to help run controlled quiz sessions, log participant choices, and analyze statistical markers of bias.

Purpose
- Support empirical experiments on **HBI detection**.  
- Record **answer order, hesitation, and switching patterns**.  
- Compute **average scores** by study level and sector.  
- Randomize difficulty distribution (3PL model) to avoid sequential bias.  
- Generate outputs for **difficulty modeling** across items.  

Project Structure
- `items_bank.json` → Bank of MCQ items with thematic + difficulty parameters.  
- `qcm_engine.py` → Core Python script (session management & bias logging).  
- `analysis/` → Scripts for modeling and detecting HBI patterns.  
- `sessions/` → Session logs and performance data.  

Usage
```bash
python qcm_engine.py
