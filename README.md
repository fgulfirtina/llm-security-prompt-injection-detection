# LLM Security & Prompt Injection Detection Model

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://llm-security-prompt-injection-detection.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B.svg)

## Overview
Large Language Models (LLMs) are highly vulnerable to adversarial attacks, specifically **Prompt Injections** and **Jailbreaks**. This project introduces a highly robust, **Two-Tier Defense Architecture** designed to detect and block malicious prompts, payload generation requests, and social engineering attacks before they reach the core LLM. 

By combining rule-based Natural Language Processing (NLP) with a fine-tuned, data-centric Deep Learning model, this Context-Aware Firewall ensures security with a low False Positive rate for legitimate technical queries.

## Key Features & Architecture

### Tier 1: Smart Heuristic Defense (Context-Aware Regex)
A fast, flexible NLP filter that acts as the first line of defense. It mitigates:
* **Payload Generation:** Blocks requests to create malware, keyloggers, and reverse shells.
* **Morphological & Insertion Evasion:** Resilient against attackers using progressive suffixes (e.g., "hacking") or inserting filler words.
* **Reverse Syntax:** Bidirectional pattern matching catches commands even when the target precedes the action.
* **SSRF & Phishing:** Prevents data exfiltration and the generation of fraudulent URLs.

### Tier 2: Context-Aware AI (Fine-Tuned DistilBERT)
Prompts that bypass Tier 1 are analyzed by a custom-trained `distilbert-base-uncased` sequence classification model.
* **Data-Centric Training:** Trained on an aggregated dataset combining malicious injection datasets (Deepset, Kaggle) with massive benign datasets (Alpaca, CodeAlpaca-20k).
* **Semantic Understanding:** The model successfully differentiates between a legitimate software engineering query (e.g., "SQL connection in Spring Boot") and an actual injection attempt.

### Post-Processing: Score Calibration
To prevent Neural Network Overconfidence, the system mathematically calibrates threat scores for safe academic research or casual conversational prompts, eliminating False Positives on benign edge cases.

## Datasets Used

To build a highly accurate and context-aware classification model, the DistilBERT architecture was fine-tuned on an aggregated dataset combining both real-world threats and complex benign queries.

**🔴 Malicious Datasets (Label 1: Threat)**
* [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections): Used to teach the model foundational manipulation tactics (e.g., "ignore previous instructions", "system prompt extraction").
* [Prompt Injection in the Wild](https://www.kaggle.com/datasets/arielzilber/prompt-injection-in-the-wild) (Kaggle): Integrated to expose the model to a wide variety of real-world adversarial attacks and jailbreak frameworks.

**🟢 Benign Datasets (Label 0: Safe)**
* [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned): Provided a baseline for standard, non-malicious user interactions and casual conversations.
* [sahil2801/CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k): **Crucial for preventing False Positives.** This dataset taught the model complex software engineering terminology, coding queries, and database configurations, ensuring that legitimate developer requests are not mistakenly flagged as malicious payloads.

## Tech Stack
* **Machine Learning:** PyTorch, HuggingFace Transformers (DistilBERT)
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Frontend/Deployment:** Streamlit
* **Logic:** Python `re` (Regular Expressions)

## About This Project
This project was developed as an advanced academic study focusing on LLM security, adversarial machine learning, and Web Application Firewall (WAF) architectures. The goal is to provide a scalable, intelligent, and context-aware solution to one of the most pressing vulnerabilities in modern AI systems.

## Author
Fatmagül Fırtına — Computer Engineering Student at Dokuz Eylul University
