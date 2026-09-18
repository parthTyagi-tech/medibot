"""
MediAssist Clinical Entity & Negation Parser
============================================
Deterministic clinical signal extraction module that inspects user queries
before triage evaluation to detect:
1. Clinical Negation (e.g. "no fever", "without chills", "denies fever", "afebrile")
2. Third-Party Entity Attribution (e.g. "my mom has cancer", "my father has lymphoma")
3. Gastrointestinal Symptoms (e.g. diarrhea, vomiting)
4. Verified non-negated active and negated symptoms extraction
"""

import re
from typing import Dict, List, Optional, Any, Tuple


class ClinicalEntityParser:
    """
    Deterministic clinical signal extractor with negation handling and subject attribution.
    """

    # Specified clinical negation patterns (clause-bounded with [^.,;!\n] to prevent punctuation bleed)
    NEGATION_PRECEDING_PATTERN = (
        r"\b(no|without|denies|not\s+having|never\s+had|zero|definitely\s+no|afebrile|negative\s+for)\b"
        r"[^.,;!\n]{0,25}\b(fever|chills|temperature|warm|pain)\b"
    )
    NEGATION_FOLLOWING_PATTERN = (
        r"\b(fever|chills)\b[^.,;!\n]{0,15}\b(is\s+absent|not\s+present|gone|resolved)\b"
    )

    CLINICAL_INTENT_WORDS = [
        r"\bpain\b", r"\bvomit\b", r"\bvomiting\b", r"\bdiarrhea\b", 
        r"\bmedication\b", r"\bmedicine\b", r"\bpill\b", r"\btablet\b", 
        r"\bsuggest\b", r"\bcure\b", r"\bremedy\b", r"\bport\b", r"\bpicc\b",
        r"\bhypertension\b", r"\basthma\b", r"\bdiabetes\b", r"\binfection\b",
        r"\bblood\s+pressure\b", r"\bheart\b", r"\bcough\b", r"\bheadache\b",
        r"\brash\b", r"\bfever\b", r"\bswelling\b", r"\bsymptom\b",
        r"\bdisease\b", r"\bcondition\b", r"\bdiagnosis\b",
        r"\bwhat\s+(is|are)\b", r"\btell\s+me\s+about\b", r"\bhow\s+to\s+(treat|manage)\b"
    ]

    # General prefix & postfix lists for broader symptom targets
    NEGATION_PREFIXES = [
        r"\b(no|without|denies|denying|negative\s+for|not\s+having|never\s+had|zero|free\s+of)\b",
        r"\b(don't\s+have|do\s+not\s+have|doesn't\s+have|does\s+not\s+have|haven't\s+had|hasn't\s+had)\b",
        r"\b(rule\s+out|definitely\s+no|absolutely\s+no)\b",
    ]
    NEGATION_POSTFIXES = [
        r"\b(none|negative|absent|resolved|denied)\b"
    ]

    # Specified third-party entity attribution pattern
    THIRD_PARTY_ATTRIBUTION_PATTERN = (
        r"\b(my|his|her|their)\s+"
        r"(mother|mom|father|dad|brother|sister|friend|child|son|daughter|wife|husband|partner|parent)\b"
        r"[\w\s]{0,35}\b(cancer|chemo|fever|leukemia|lymphoma|tumor)\b"
    )

    # Relative vocabulary
    THIRD_PARTY_RELATIVES = (
        r"mother|mom|mum|father|dad|brother|sister|friend|child|kid|son|daughter|"
        r"wife|husband|partner|parent|grandma|grandmother|grandpa|grandfather|"
        r"uncle|aunt|cousin|colleague|coworker|neighbor"
    )

    # Symptom vocabulary
    SYMPTOM_PATTERNS = {
        "fever": r"\b(fever|chills|shivering|temperature|temp|febrile|high\s+temp)\b",
        "cold": r"\b(cold|cough|coughing|runny\s+nose|congestion|congested|sneezing|sore\s+throat)\b",
        "headache": r"\b(headache|head\s+pain|migraine)\b",
        "pain": r"\b(pain|ache|aching|stomach\s+pain|abdominal\s+pain|belly\s+pain|cramps)\b",
        "diarrhea": r"\b(diarrhea|loose\s+stools?|watery\s+stools?)\b",
        "vomiting": r"\b(vomiting|vomit|threw\s+up|throwing\s+up|nausea|nauseous)\b",
        "shortness_of_breath": r"\b(shortness\s+of\s+breath|difficulty\s+breathing|trouble\s+breathing|dyspnea)\b",
        "chest_pain": r"\b(chest\s+pain|pressure\s+in\s+chest|tightness\s+in\s+chest)\b"
    }

    CANCER_PATTERNS = r"\b(cancer|chemo|chemotherapy|oncology|oncologist|leukemia|lymphoma|carcinoma|tumor|tumour|malignancy|radiation\s+therapy)\b"

    @classmethod
    def is_negated(cls, text: str, target_pattern: str, window_chars: int = 25) -> bool:
        """
        Determines whether a clinical target is explicitly negated in the given text.
        Inspects pre-target (up to 25 chars) and post-target (up to 15 chars) grammatical windows.
        """
        lower_text = text.lower()

        # Specific medical terms that imply negation
        if "afebrile" in lower_text and ("fever" in target_pattern or "temp" in target_pattern):
            return True

        target_matches = list(re.finditer(target_pattern, lower_text))
        if not target_matches:
            return False

        for match in target_matches:
            start_pos, end_pos = match.span()

            # Preceding window check (25 chars)
            pre_start = max(0, start_pos - 25)
            pre_window = lower_text[pre_start:start_pos]
            combined_pre = pre_window + match.group(0)

            # Check specified preceding negation pattern for fever/chills/pain
            if re.search(cls.NEGATION_PRECEDING_PATTERN, combined_pre):
                return True

            for prefix in cls.NEGATION_PREFIXES:
                if re.search(prefix, pre_window):
                    return True

            # Following window check (15 chars)
            post_end = min(len(lower_text), end_pos + 15)
            post_window = lower_text[end_pos:post_end]
            combined_post = match.group(0) + post_window

            # Check specified following negation pattern
            if re.search(cls.NEGATION_FOLLOWING_PATTERN, combined_post):
                return True

            for postfix in cls.NEGATION_POSTFIXES:
                if re.search(r"^\s*[:\-=]?\s*" + postfix, post_window):
                    return True

        return False

    @classmethod
    def attribute_subject(cls, text: str, target_pattern: str) -> Tuple[bool, Optional[str]]:
        """
        Determines whether a condition or symptom pertains to 'self', a 'third_party', or None.
        """
        lower_text = text.lower()
        if not re.search(target_pattern, lower_text):
            return False, None

        # Check explicit specified third-party pattern first
        if re.search(cls.THIRD_PARTY_ATTRIBUTION_PATTERN, lower_text):
            # Check if this specific target is in the third-party phrase
            tp_specific = rf"\b(my|his|her|their)\s+({cls.THIRD_PARTY_RELATIVES})\b[\w\s]{{0,35}}{target_pattern}"
            if re.search(tp_specific, lower_text):
                return True, "third_party"

        # Check for reverse attribution (e.g. "fever in my mother", "cancer for his dad")
        tp_reverse = rf"{target_pattern}[\w\s]{{0,25}}\b(in|for|with)\s+(my|his|her|their)\s+({cls.THIRD_PARTY_RELATIVES})\b"
        if re.search(tp_reverse, lower_text):
            return True, "third_party"

        # Standalone relative subject clause (e.g. "my mom has leukemia", "dad was diagnosed with cancer")
        tp_clause = rf"\b(my\s+)?({cls.THIRD_PARTY_RELATIVES})\s+(has|have|is\s+having|was\s+diagnosed|got)\b[\w\s]{{0,35}}{target_pattern}"
        if re.search(tp_clause, lower_text):
            return True, "third_party"

        # Self disclosure patterns (e.g. "I have cancer", "diagnosed with leukemia", "my cancer")
        self_pattern = rf"\b(i\s+have|i\s+was|i'm|i\s+am|diagnosed\s+with|my\s+oncologist|my\s+chemo|my\s+cancer)\b[\w\s]{{0,35}}{target_pattern}"
        if re.search(self_pattern, lower_text) or re.search(rf"{target_pattern}[\w\s]{{0,20}}\b(i\s+have|diagnosed\s+in\s+me)\b", lower_text):
            return True, "self"

        # If third-party relative mentioned anywhere in a short query and no self-reference
        if re.search(rf"\b({cls.THIRD_PARTY_RELATIVES})\b", lower_text) and not re.search(r"\b(i\s+have|i'm|i\s+am)\b", lower_text):
            return True, "third_party"

        # Default to self if target is present and no third-party indicators exist
        return True, "self"

    @classmethod
    def extract_clinical_signals(cls, text: str) -> Dict[str, Any]:
        """
        Extracts verified clinical signals from user text.

        Returns:
        {
            "has_cancer": bool,
            "cancer_subject": "self" | "third_party" | None,
            "has_fever": bool,
            "fever_negated": bool,
            "fever_subject": "self" | "third_party" | None,
            "has_gi_symptoms": bool,
            "gi_symptoms": List[str],
            "active_symptoms": List[str],
            "negated_symptoms": List[str]
        }
        """
        lower_text = (text or "").lower()

        # 1. Cancer signals & attribution
        has_cancer, cancer_subject = cls.attribute_subject(lower_text, cls.CANCER_PATTERNS)

        # 2. Fever signals & negation
        has_fever_mention = bool(re.search(cls.SYMPTOM_PATTERNS["fever"], lower_text))
        fever_negated = False
        fever_subject = None

        if has_fever_mention:
            fever_negated = cls.is_negated(lower_text, cls.SYMPTOM_PATTERNS["fever"])
            _, fever_subject = cls.attribute_subject(lower_text, cls.SYMPTOM_PATTERNS["fever"])
        elif "afebrile" in lower_text:
            has_fever_mention = True
            fever_negated = True
            fever_subject = "self"

        has_fever = has_fever_mention and not fever_negated

        # 3. GI symptoms (diarrhea, vomiting, nausea)
        gi_symptoms = []
        if re.search(cls.SYMPTOM_PATTERNS["diarrhea"], lower_text) and not cls.is_negated(lower_text, cls.SYMPTOM_PATTERNS["diarrhea"]):
            gi_symptoms.append("diarrhea")
        if re.search(cls.SYMPTOM_PATTERNS["vomiting"], lower_text) and not cls.is_negated(lower_text, cls.SYMPTOM_PATTERNS["vomiting"]):
            gi_symptoms.append("vomiting")
        has_gi_symptoms = len(gi_symptoms) > 0

        # 4. Active (non-negated) and negated symptoms lists
        active_symptoms = []
        negated_symptoms = []

        for symptom_name, pattern in cls.SYMPTOM_PATTERNS.items():
            if re.search(pattern, lower_text):
                if cls.is_negated(lower_text, pattern):
                    negated_symptoms.append(symptom_name)
                else:
                    active_symptoms.append(symptom_name)
            elif symptom_name == "fever" and "afebrile" in lower_text:
                negated_symptoms.append("fever")

        # 5. Third-party relative relationship detection
        rel_match = re.search(rf"\b(my|his|her|their)\s+({cls.THIRD_PARTY_RELATIVES})\b", lower_text)
        relative_relation = rel_match.group(2) if rel_match else None
        is_tp = (cancer_subject == "third_party") or (fever_subject == "third_party") or (relative_relation is not None)

        # 6. Overall Clinical Signal Check (Used to block mixed-intent greeting hijacking)
        has_clinical_keywords = any(bool(re.search(w, lower_text)) for w in cls.CLINICAL_INTENT_WORDS)
        has_clinical_signals = has_fever or has_cancer or has_gi_symptoms or has_clinical_keywords

        return {
            "has_cancer": has_cancer,
            "cancer_subject": cancer_subject,
            "has_fever": has_fever,
            "fever_negated": fever_negated,
            "fever_subject": fever_subject,
            "has_gi_symptoms": has_gi_symptoms,
            "gi_symptoms": gi_symptoms,
            "active_symptoms": active_symptoms,
            "negated_symptoms": negated_symptoms,
            "has_clinical_signals": has_clinical_signals,
            # Preserved for backward compatibility
            "extracted_symptoms": active_symptoms,
            "is_third_party_query": is_tp,
            "relative_relation": relative_relation
        }
