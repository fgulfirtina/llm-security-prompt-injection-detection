import streamlit as st
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import time
import re

# ==========================================
# CONFIG — tune these without retraining
# ==========================================
SAFE_THRESHOLD      = 0.30   # Below this → allow
BLOCK_THRESHOLD     = 0.65   # Above this → block
# Between SAFE and BLOCK → abstain (ask for clarification)
MODEL_DIR = "./distilbert_context_aware_model"


import re

# ==========================================
# TIER 1 — SURGICAL REGEX
#
# Design principle: every pattern must match BOTH:
#   (a) a harmful *verb* (create, write, generate, bypass, ignore…)
#   (b) a harmful *object* (malware, SQL injection, system prompt…)
#
# This stops the engine from blocking "how does SQL injection work?"
# (informational) while still blocking "write me SQL injection code"
# (action request).
# ==========================================

# Harmful verbs
_GENERATE = r"(?:writ(?:e|es|ing|ten)|creat(?:e|es|ed|ing)|build(?:s|ing)?|cod(?:e|es|ed|ing)|develop(?:s|ed|ing)?|generat(?:e|es|ed|ing)|produc(?:e|es|ed|ing)|craft(?:s|ed|ing)?|output(?:s|ting)?|giv(?:e|es|ing)\s+me|show(?:s|ing)?\s+me|provid(?:e|es|ed|ing))"
_PERFORM  = r"(?:perform(?:s|ed|ing)?|execut(?:e|es|ed|ing)|inject(?:s|ed|ing)?|conduct(?:s|ed|ing)?|run(?:s|ning)?|launch(?:es|ed|ing)?|carry(?:ing)?\s+out|carried\s+out)"
_BYPASS   = r"(?:bypass(?:es|ed|ing)?|circumvent(?:s|ed|ing)?|disabl(?:e|es|ed|ing)|overrid(?:e|es|ing)|evad(?:e|es|ed|ing)|get(?:s|ting)?\s+around|break(?:s|ing)?)"
_EXFIL    = r"(?:send(?:s|ing)?|transmit(?:s|ting|ted)?|post(?:s|ed|ing)?|upload(?:s|ed|ing)?|exfiltrat(?:e|es|ed|ing)|dump(?:s|ed|ing)?|leak(?:s|ed|ing)?)"

#Specific action verbs
_MANIPULATE = r"(?:manipulat(?:e|es|ed|ing)|modif(?:y|ies|ied|ying)|alter(?:s|ed|ing)?|inject(?:s|ed|ing)?\s+into|tamper(?:s|ed|ing)?\s+with)"
_RECON_VERB = r"(?:discover(?:s|ed|ing)?|scan(?:s|ned|ning)?\s+for|find(?:s|ing)?|enumerat(?:e|es|ed|ing)|search(?:es|ed|ing)?\s+for|list(?:s|ed|ing)?)"

# Harmful objects
_MALWARE  = r"(?:keylog(?:ger|ging)?|virus|ransomware|malware|trojan|rootkit|spyware|worm|backdoor|reverse\s+shell|botnet|shellcode|exploit\s+(?:code|script)|payload(?:s)?)"
_ATTACK   = r"(?:sql\s+injection|sqli|xss|cross.site\s+scripting|csrf|ssrf|rce|remote\s+code\s+execution|buffer\s+overflow|privilege\s+escalation|directory\s+traversal|path\s+traversal|command\s+injection)"
_PHISH    = r"(?:phish(?:ing|es|ed)?|fake\s+login\s+page(?:s)?|credential\s+harvest(?:ing)?|account\s+takeover\s+page(?:s)?)"
_EXFOBJ   = r"(?:system\s+prompt(?:s)?|conversation\s+history|context\s+window|internal\s+instruction(?:s)?|prior\s+message(?:s)?)"

# Specific target objects
_DB_QUERY   = r"(?:database\s+)?(?:sql\s+)?quer(?:y|ies)"
_RECON_OBJ  = r"(?:all\s+)?(?:open|active|vulnerable)\s+port(?:s)?"


TIER1_PATTERNS = [
    # Payload generation: verb + malware/exploit object
    rf"{_GENERATE}.*?{_MALWARE}",
    rf"{_MALWARE}.*?{_GENERATE}",          # reverse order ("keylogger code, write it")

    # Attack execution: verb + attack type
    rf"{_PERFORM}.*?{_ATTACK}",

    # Phishing / social engineering generation
    rf"{_GENERATE}.*?{_PHISH}",

    # Safety bypass: explicit bypass verb + safety target
    rf"{_BYPASS}.*?(?:safety|restriction(?:s)?|filter(?:s)?|guardrail(?:s)?|polic(?:y|ies)|rule(?:s)?|content\s+polic(?:y|ies)|moderation)",

    # Prompt injection markers (these have no benign interpretation)
    r"ignor(?:e|es|ed|ing)\s+(?:all\s+)?(?:previous|prior|above|your)\s+(?:instruction(?:s)?|rule(?:s)?|guideline(?:s)?|direction(?:s)?|prompt(?:s)?)",
    r"disregard(?:s|ed|ing)?\s+(?:all\s+)?(?:previous|prior|above|your)\s+(?:instruction(?:s)?|rule(?:s)?|guideline(?:s)?|direction(?:s)?)",
    r"forget(?:s|ting)?\s+(?:all\s+)?(?:previous|prior|above|your)\s+(?:instruction(?:s)?|rule(?:s)?|guideline(?:s)?|direction(?:s)?)",
    r"you\s+are\s+now\s+(?:in\s+)?(?:developer|god|unrestricted|jailbreak|dan)\s+mode",
    r"\bdo\s+anything\s+now\b",
    r"\bdeveloper\s+mode\s+enabled\b",
    r"\bjailbreak(?:ed)?\b.*?(?:mode|enabled|activated)",

    # Indirect injection: fetch external instructions
    r"(?:load(?:s|ed|ing)?|read(?:s|ing)?|fetch(?:es|ed|ing)?|retriev(?:e|es|ed|ing)|visit(?:s|ed|ing)?|access(?:es|ed|ing)?)\s+.*?(?:instruction(?:s)?|rule(?:s)?|guideline(?:s)?|command(?:s)?)\s+from\s+(?:http|www\.|this\s+url|the\s+link)",

    # Data exfiltration: send model internals somewhere
    rf"{_EXFIL}.*?{_EXFOBJ}",
    rf"{_EXFIL}.*?(?:your\s+(?:output|response|reply))\s+to\s+(?:http|www\.|this\s+url)",

    # ==========================================
    # ETHICS, SAFETY OVERRIDES & RECON
    # ==========================================
    
    # "Without safety", "do not care about safety" (flexible spacing and phrasing)
    r"(?:without|ignor(?:e|es|ed|ing)|disregard(?:s|ed|ing)?|do(?:es)\s+not\s+care\s+about|do(?:es)n['’]?t\s+care\s+about|remov(?:e|es|ed|ing)|drop(?:s|ping)?).*?(?:any\s+)?(?:safety|security)\s+(?:restriction(?:s)?|filter(?:s)?|rule(?:s)?|guideline(?:s)?|protocol(?:s)?|guardrail(?:s)?)",

    # "Reverse ethics", "ignore core ethics" (catches deep persona manipulations)
    r"(?:revers(?:e|es|ed|ing)|ignor(?:e|es|ed|ing)|disregard(?:s|ed|ing)?|bypass(?:es|ed|ing)?|do\s+not\s+care\s+about|don['’]?t\s+care\s+about|suspend(?:s|ed|ing)?).*?(?:your\s+)?(?:core\s+)?(?:ethics|ethical\s+(?:guideline(?:s)?|rule(?:s)?|filter(?:s)?|protocol(?:s)?|principle(?:s)?))",

    # "Manipulate database query" (stops subtle SQLi logic requests)
    rf"{_MANIPULATE}.*?{_DB_QUERY}",

    # "Discover open ports" (stops network reconnaissance scripts like our sci-fi villain)
    rf"{_RECON_VERB}.*?{_RECON_OBJ}"
]

_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in TIER1_PATTERNS]


def check_tier1(prompt: str):
    """Returns (is_malicious: bool, matched_pattern: str | None)"""
    for pat, raw in zip(_COMPILED, TIER1_PATTERNS):
        if pat.search(prompt):
            return True, raw
    return False, None


# ==========================================
# TIER 2 — DISTILBERT MODEL
# ==========================================
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()


def score_prompt(prompt: str) -> float:
    """Returns P(malicious) in range [0, 1]."""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
    return probs[0][1].item()


# ==========================================
# UI
# ==========================================
st.set_page_config(
    page_title="Prompt Injection Detector",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ LLM Security: Prompt Injection Detector")
st.markdown(
    "A two-tier firewall combining **surgical intent-based regex (Tier 1)** "
    "and a **fine-tuned DistilBERT classifier (Tier 2)** with a three-zone "
    "decision router. Tier 1 blocks only explicit action+object pairs. "
    "Tier 2 handles semantic attacks. Borderline scores trigger abstention "
    "instead of guessing."
)
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Chat interface")
    user_prompt = st.text_area("Enter a prompt:", height=150, placeholder="Ask anything…")
    submit_btn = st.button("Submit 🚀")
    chat_box = st.empty()

with col2:
    st.subheader("Security analysis")
    analysis_box = st.empty()


# ==========================================
# DECISION LOGIC
# ==========================================
if submit_btn and user_prompt.strip():
    with st.spinner("Analysing…"):
        time.sleep(0.3)

        # --- Tier 1 ---
        blocked_t1, matched = check_tier1(user_prompt)

        if blocked_t1:
            with analysis_box.container():
                st.error("🚨 Tier 1 — BLOCKED")
                st.markdown(f"**Matched pattern:**")
                st.code(matched, language="text")
                st.progress(100)
                st.caption("100% malicious — explicit intent rule")

            with chat_box.container():
                st.error("🚨 Security violation (Tier 1)")
                st.warning(
                    "Your request matched an explicit malicious intent rule "
                    "and has been blocked before reaching the model."
                )

        else:
            # --- Tier 2 ---
            score = score_prompt(user_prompt)
            pct_mal = score * 100
            pct_safe = (1 - score) * 100

            with analysis_box.container():
                st.success("✅ Tier 1 — passed")

                st.write("**Tier 2 — safe probability**")
                st.progress(int(pct_safe))
                st.caption(f"{pct_safe:.1f}%")

                st.write("**Tier 2 — malicious probability**")
                st.progress(int(pct_mal))
                st.caption(f"{pct_mal:.1f}%")

                # Colour-coded verdict badge
                if score < SAFE_THRESHOLD:
                    st.success(f"Verdict: **ALLOW** (score {pct_mal:.1f}% < {SAFE_THRESHOLD*100:.0f}% threshold)")
                elif score < BLOCK_THRESHOLD:
                    st.warning(
                        f"Verdict: **REVIEW** ({pct_mal:.1f}% — between "
                        f"{SAFE_THRESHOLD*100:.0f}% and {BLOCK_THRESHOLD*100:.0f}%)"
                    )
                else:
                    st.error(f"Verdict: **BLOCK** (score {pct_mal:.1f}% ≥ {BLOCK_THRESHOLD*100:.0f}% threshold)")

            with chat_box.container():
                if score < SAFE_THRESHOLD:
                    st.success("✅ Safe request — forwarding to LLM…")
                    st.info("🤖 **LLM response:** *(your LLM's reply would appear here)*")

                elif score < BLOCK_THRESHOLD:
                    st.warning("⚠️ Uncertain intent — request held for review")
                    st.info(
                        "Our classifier is not confident about this request. "
                        "Could you rephrase or provide more context about what you need?"
                    )

                else:
                    st.error("🚨 Security violation (Tier 2)")
                    st.warning(
                        "Our AI classifier flagged this request as likely malicious "
                        "with high confidence. If you believe this is a mistake, "
                        "please rephrase your question."
                    )

elif submit_btn:
    st.warning("Please enter a prompt first.")


# ==========================================
# THRESHOLD TUNING GUIDE (collapsible)
# ==========================================
with st.expander("⚙️ Threshold tuning guide"):
    st.markdown("""
| Variable | Current | Effect of lowering | Effect of raising |
|---|---|---|---|
| `SAFE_THRESHOLD` | 0.30 | More requests go to review zone | Fewer false alarms, more attacks slip through |
| `BLOCK_THRESHOLD` | 0.65 | More requests blocked outright | Fewer false alarms, more attacks in review zone |

After training, open `distilbert_injection_detector/threshold_analysis.csv` to see
the exact precision/recall tradeoff at each threshold point and pick the row that
matches your security requirement.

**Rule of thumb:**
- High-security internal tool → lower both thresholds
- Public-facing chatbot → raise `SAFE_THRESHOLD` slightly, keep `BLOCK_THRESHOLD` at 0.65
""")


# ==========================================
# FOOTER
# ==========================================
st.divider()
st.markdown(
    """
<div style="text-align:center;color:gray;padding:10px">
    <b>Developed by Fatmagül Fırtına</b><br>
    Final Year Computer Engineering Student |
    <a href="https://github.com/fgulfirtina" target="_blank">GitHub</a> |
    <a href="https://linkedin.com/in/fatmagul-firtina" target="_blank">LinkedIn</a><br><br>
    <i>Advanced LLM Web Application Firewall — hybrid regex + DistilBERT approach
    with three-zone decision routing and confidence-based abstention.</i>
</div>
""",
    unsafe_allow_html=True,
)
