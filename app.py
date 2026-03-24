import streamlit as st
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import time
import re

# ==========================================
# TIER 1: SMART HEURISTIC DEFENSE (FLEXIBLE REGEX)
# ==========================================
MALICIOUS_PATTERNS = [
    # Prompt Injection & Jailbreak 
    r"(ignore|forget|disregard).*?(previous|all).*?(instructions|rules|directions|guidelines)",
    r"system\s+prompt",
    r"developer\s+mode",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"bypass.*?(restrictions|filters|safety|firewall|authentication|auth)",
    
    # Payload Generation (Morphological Evasion Korumalı: keylog(ger|ging), hack(ing) vb.)
    r"(write|create|build|code|develop|generate|give|show|provide|produce|output|craft|design).*?(keylog(ger|ging)?|virus|ransomware|malware|trojan|rootkit|exploit|hack(ing)?\s+(code|script)|(code|script)\s+to\s+hack(ing)?|payload|shellcode)",
    r"(write|create|build|code|develop|generate|give|show|provide|produce|output|craft|design).*?(reverse\s+shell|backdoor|botnet)",
    r"(perform|execute|inject|write|create|give|show|provide|conduct).*?(sql\s+injection|sqli)",
    r"(perform|execute|inject|write|create|give|show|provide|conduct).*?(cross-site\s+scripting|xss)",
    
    # Phishing, Fake Logins & Malicious Redirections
    r"(write|create|generate|design|send|redirect|route|give|show|provide).*?(phish(ing)?|fake\s+login|credential\s+harvesting)",
    r"(create|generate|give|show|provide).*?(malicious\s+link|scam\s+page|fraudulent\s+url)",
    r"trick\s+users\s+into",
    
    # SSRF, Data Exfiltration & Indirect Prompt Injection
    r"(visit|access|fetch|read|retrieve|go\s+to).*?(link|url|website|http[s]?://|www\.).*?(instructions|commands|prompt|payload)",
    r"(send|transmit|post|upload|dump|exfiltrate).*?(output|response|data|summary|chat|credentials|passwords).*?(to|http[s]?://|www\.)",
    r"load.*?(rules|instructions|guidelines).*?(from|http[s]?://|www\.)",
    
    # Tutorials & How-To
    r"(how\s+to|guide|tutorial|steps).*?(hack(ing)?|bypass(ing)?|exploit(ing)?|keylog(ger|ging)?|ddos(ing)?|phish(ing)?|breach(ing)?)",
    r"(how\s+to|guide|tutorial|steps).*?(create|write|build|code|give|show|provide).*?(keylog(ger|ging)?|virus|malware|ransomware|trojan|exploit|payload)",

    # Reverse Order Attacks (Master Yoda Attacks)
    r"(keylog(ger|ging)?|virus|ransomware|malware|trojan|rootkit|exploit|hack(ing)?\s+(code|script)|(code|script)\s+to\s+hack(ing)?|payload|shellcode).*?(write|create|build|code|develop|generate|give|show|provide|produce|output|craft|design)",
    r"(reverse\s+shell|backdoor|botnet).*?(write|create|build|code|develop|generate|give|show|provide|produce|output|craft|design)",
    r"(sql\s+injection|sqli|cross-site\s+scripting|xss).*?(perform|execute|inject|write|create|give|show|provide|conduct)"
]

def check_tier1_heuristics(prompt):
    prompt_lower = prompt.lower()
    for pattern in MALICIOUS_PATTERNS:
        if re.search(pattern, prompt_lower):
            return True, pattern 
    return False, None

# ==========================================
# POST-PROCESSING: FALSE POSITIVE CALIBRATION
# ==========================================
# 1. Academic & Scenario Traps
FP_TRIGGERS = [r"\bresearch\b", r"\bact as\b", r"\bscenario\b", r"\beducational\b", r"\bproject\b", r"\bthesis\b", r"\bpretend\b"]

# 2. Casual Chat & Daily Conversations
CASUAL_TRIGGERS = [r"\bhello\b", r"\bhi\b", r"\bgood morning\b", r"\bbeautiful day\b", r"\bhow are you\b", r"\bthanks\b", r"\bthank you\b", r"\bhey\b", r"\bweather\b", r"\bday\b", r"\bnight\b", r"\bchat\b"]

# Danger Keywords (If these exist, DO NOT calibrate the score! Trust the AI.)
DANGER_KEYWORDS = [r"\bhack", r"\bscript\b", r"\bcode\b", r"\bpassword\b", r"\bdatabase\b", r"\bsql\b", r"\bserver\b", r"\bnetwork\b", r"\bpayload\b", r"\bbypass\b", r"\bignore\b", r"\bprompt\b", r"\binstructions\b"]

def calibrate_score(prompt_lower, prob_malicious):
    """Calibrates the AI's malicious score to prevent False Positives on benign contexts."""
    has_danger = any(re.search(d, prompt_lower) for d in DANGER_KEYWORDS)
    
    # If there is a direct danger keyword, do not lower the score.
    if has_danger:
        return prob_malicious, False, ""

    is_trap = any(re.search(t, prompt_lower) for t in FP_TRIGGERS)
    is_casual = any(re.search(c, prompt_lower) for c in CASUAL_TRIGGERS)
    is_short_text = len(prompt_lower) < 60 # Short texts without danger words are usually safe
    
    # Rule 1: Research and Academic Context
    if is_trap and prob_malicious > 50.0:
        return prob_malicious / 3.0, True, "Calibrated: Educational/Research Context"
        
    # Rule 2: Casual Conversation & Small Talk
    if (is_casual or is_short_text) and prob_malicious > 50.0:
        return prob_malicious / 4.0, True, "Calibrated: Casual Conversation / Short Text"
        
    return prob_malicious, False, ""

# ==========================================
# TIER 2: DISTILBERT CLASSIFICATION MODEL
# ==========================================
@st.cache_resource
def load_model():
    save_directory = "./distilbert_context_aware_model" 
    tokenizer = DistilBertTokenizer.from_pretrained(save_directory)
    model = DistilBertForSequenceClassification.from_pretrained(save_directory)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
MALICIOUS_THRESHOLD = 50.0 

# ==========================================
# UI LAYOUT
# ==========================================
# st.set_page_config en üstte (importlardan hemen sonra) olmalı
st.set_page_config(page_title="Prompt Injection Detector", page_icon="🛡️", layout="wide")

st.title("🛡️ LLM Security: Prompt Injection Detection")
st.markdown("This system evaluates user prompts to protect Large Language Models (LLMs) from adversarial attacks. It combines **Intent-Based NLP Pattern Matching (Tier 1)**, a **Context-Aware DistilBERT Model (Tier 2)**, and **Score Calibration** to ensure maximum security with minimal false positives.")

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Chat Interface")
    user_prompt = st.text_area("Ask a question or give a command to the LLM:", height=150)
    submit_btn = st.button("Submit 🚀")
    
    # Uyarıların ve LLM cevabının burada çıkması için boş bir alan ayırıyoruz
    chat_result_box = st.empty() 

with col2:
    st.subheader("Security Analysis")
    analysis_box = st.empty()

# ==========================================
# LOGIC & EXECUTION
# ==========================================
if submit_btn and user_prompt:
    with st.spinner("Two-tier firewall is analyzing the request..."):
        time.sleep(0.5)
        
        # TIER 1
        is_malicious_tier1, matched_pattern = check_tier1_heuristics(user_prompt)
        
        if is_malicious_tier1:
            with analysis_box.container():
                st.error("🚨 **TIER 1: NLP ENGINE BLOCKED!**")
                st.write(f"**Matched Pattern:** `{matched_pattern}`")
                st.progress(100)
                st.write("100.00% Malicious (Action-Based Rule)")
            
            # Sonucu SOL kolondaki (Chat) alana yazdırıyoruz ki ekrana sığsın
            with chat_result_box.container():
                st.error("🚨 **SECURITY VIOLATION (TIER 1)** 🚨")
                st.warning("Your request contains direct malicious actions or rule violations. It has been blocked by the heuristic engine.")
            
        else:
            # TIER 2
            inputs = tokenizer(user_prompt, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=1)
                raw_prob_malicious = probabilities[0][1].item() * 100
                
                # Apply Score Calibration
                prob_malicious, was_calibrated, calib_message = calibrate_score(user_prompt.lower(), raw_prob_malicious)
                prob_benign = 100.0 - prob_malicious
                
                is_malicious_tier2 = True if prob_malicious >= MALICIOUS_THRESHOLD else False
            
            with analysis_box.container():
                st.success("✅ **TIER 1: PASSED (No Action Patterns)**")
                
                st.write("**TIER 2: AI Benign Probability:**")
                st.progress(int(prob_benign))
                st.write(f"{prob_benign:.2f}%")
                
                st.write("**TIER 2: AI Malicious Probability:**")
                st.progress(int(prob_malicious))
                st.write(f"{prob_malicious:.2f}%")
                
                if was_calibrated:
                    st.info(f"ℹ️ *{calib_message}*")
                else:
                    st.caption(f"*(Strict Block Threshold: {MALICIOUS_THRESHOLD}%)*")

            # Sonucu SOL kolondaki (Chat) alana yazdırıyoruz
            with chat_result_box.container():
                if is_malicious_tier2:
                    st.error("🚨 **SECURITY VIOLATION (TIER 2)** 🚨")
                    st.warning("Our AI model classified your intent as malicious. Request blocked.")
                else:
                    st.success("✅ **Safe Request! Passed both tiers.** Forwarding to the LLM...")
                    st.info(f"**🤖 LLM Response:** \n\n(A normal AI model's response for your safe question will be displayed here.)")

# ==========================================
# FOOTER / ABOUT SECTION
# ==========================================
st.divider()

st.markdown("""
<div style="text-align: center; color: gray; padding: 10px;">
    <b>Developed by Fatmagül Fırtına</b><br>
    Final Year Computer Engineering Student | <a href="https://github.com/fgulfirtina" target="_blank">GitHub</a> | <a href="https://linkedin.com/in/fatmagul-firtina" target="_blank">LinkedIn</a><br>
    <br>
    <i>This project demonstrates an advanced Web Application Firewall (WAF) architecture for Large Language Models. 
    It prevents prompt injections, payload generation, and jailbreak attempts using a hybrid approach of Regex Heuristics and a fine-tuned DistilBERT model.</i>
</div>
""", unsafe_allow_html=True)