"""
Auto-learned rules from LLM responses.
This module tracks LLM patterns and generates regex rules automatically.
"""
import re
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from datetime import datetime

LEARNED_RULES_FILE = "rules/learned_rules.json"

def load_learned_rules() -> List[Dict]:
    """Load learned rules from file."""
    if not os.path.exists(LEARNED_RULES_FILE):
        return []
    try:
        with open(LEARNED_RULES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("rules", [])
    except Exception:
        return []

def save_learned_rules(rules: List[Dict]):
    """Save learned rules to file."""
    os.makedirs(os.path.dirname(LEARNED_RULES_FILE), exist_ok=True)
    data = {
        "version": 1,
        "last_updated": datetime.now().isoformat(),
        "rules": rules
    }
    with open(LEARNED_RULES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_pattern_with_llm(examples: List[str], offer_type: str, unit: Optional[str] = None) -> Optional[str]:
    """
    Use LLM to generate a regex pattern from example texts.
    This produces more accurate and flexible patterns than regex manipulation.
    """
    try:
        import google.generativeai as genai
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={
                "temperature": 0.3,
                "response_mime_type": "application/json",
            },
        )
        
        prompt = f"""You are a regex pattern generator. Analyze these examples of promotional text and generate a SINGLE regex pattern that matches all of them.

Examples (all are {offer_type} offers):
{chr(10).join(f"- {ex}" for ex in examples[:5])}

Requirements:
1. The pattern must match ALL examples above
2. Make numbers flexible (use \\d+(?:\\.\\d+)?)
3. Make currency flexible ([$£]?)
4. Make common words optional (up to, on, off, etc.)
5. Make whitespace flexible (\\s+)
6. Return ONLY the regex pattern string, nothing else
7. Pattern should be general enough to match variations but specific enough to avoid false positives

Return JSON:
{{
  "pattern": "your regex pattern here",
  "explanation": "brief explanation"
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Handle JSON or plain text response
        if result_text.startswith("{"):
            result = json.loads(result_text)
            pattern = result.get("pattern", "").strip()
        else:
            # LLM might return just the pattern
            pattern = result_text.strip()
            # Remove markdown code blocks if present
            pattern = re.sub(r'^```(?:regex|python)?\s*', '', pattern)
            pattern = re.sub(r'```\s*$', '', pattern)
            pattern = pattern.strip().strip('"').strip("'")
        
        # Validate pattern is reasonable
        if pattern and len(pattern) < 300:
            try:
                # Remove leading/trailing quotes
                pattern = pattern.strip().strip('"').strip("'")
                re.compile(pattern)  # Test if pattern compiles
                return pattern
            except re.error as e:
                # Pattern doesn't compile, fallback
                return None
        
        return None
    except Exception as e:
        # Fallback to simple pattern extraction
        if examples:
            return extract_pattern_from_text_simple(examples[0], offer_type, unit)
        return None

def extract_pattern_from_text_simple(text: str, offer_type: str, value: Optional[float] = None, unit: Optional[str] = None) -> Optional[str]:
    """
    Generate a regex pattern from a text example.
    Returns None if pattern cannot be reliably extracted.
    """
    if not text or len(text) < 5:
        return None
    
    # Start with escaped text
    pattern = re.escape(text)
    
    # Replace ALL numbers with flexible patterns
    pattern = re.sub(r'\\d+(?:\\.\\d+)?', r'\\d+(?:\\.\\d+)?', pattern)
    
    # Make currency flexible - handle both $ and £
    if r'\$' in pattern or r'£' in pattern:
        pattern = re.sub(r'\\[\\$£]', r'[\\$£]?', pattern)
    
    # Make percentages flexible
    if r'\%' in pattern:
        pattern = pattern.replace(r'\%', r'%?')
    
    # Make common words flexible
    # Handle "up to" variations
    pattern = pattern.replace(r'up\ to', r'(?:up\s+to|up\s+|)?')
    pattern = pattern.replace(r'Up\ To', r'(?:[Uu]p\s+[Tt]o|[Uu]p\s+|)?')
    
    # Make "on", "off", "under", "below" optional
    for word in ["on", "off", "under", "below", "over", "above"]:
        escaped = re.escape(word)
        if escaped in pattern:
            pattern = pattern.replace(escaped, f'(?:{escaped}\\s+)?')
    
    # Simplify - remove overly complex patterns
    if len(pattern) > 150:
        # Try a simpler version focusing on key parts
        words = text.split()
        if len(words) >= 3:
            # Take first 3 words + any numbers/currency
            key_parts = words[:3]
            simple_pattern = "\\s+".join([re.escape(w) for w in key_parts])
            if re.search(r'[\d$£%]', text):
                simple_pattern = simple_pattern.replace(r'\$', r'[\\$£]?')
                simple_pattern = simple_pattern.replace(r'£', r'[\\$£]?')
                simple_pattern = re.sub(r'\\d+', r'\\d+(?:\\.\\d+)?', simple_pattern)
            return simple_pattern
    
    return pattern

def generate_rule_function(pattern: str, offer_type: str, value_template: Optional[str] = None, unit: Optional[str] = None, priority: int = 90) -> str:
    """Generate Python code for a rule function."""
    pattern_var = f"PATTERN_{offer_type.upper()}_{priority}"
    func_name = f"parse_learned_{offer_type}_{priority}"
    
    # Generate value extraction
    value_extract = ""
    if value_template:
        if "%" in value_template:
            value_extract = 'pm = re.search(r"(?i)(\\d+(?:\\.\\d+)?)\\s*%", s)\n        if pm:\n            value = float(pm.group(1))'
        elif "$" in value_template or "£" in value_template:
            value_extract = 'mm = re.search(r"(?i)[$£]\\s*(\\d+(?:\\.\\d+)?)", s)\n        if mm:\n            value = float(mm.group(1))'
    
    code = f'''
def {func_name}(raw: str) -> Dict[str, Any]:
    """Auto-learned rule for {offer_type}."""
    s = (raw or "").strip()
    if not re.search(r"{pattern}", s, re.IGNORECASE):
        return {{'offer_type': 'unspecified', 'raw': raw}}
    
    result = {{
        'offer_type': '{offer_type}',
        'value': None,
        'unit': '{unit or ""}',
        'modifier': [],
        'stackable': 'unknown',
        'language': 'en',
        'raw': raw
    }}
    
    {value_extract if value_extract else "# No value extraction"}
    
    return result
'''
    return code.strip()

def analyze_llm_patterns(llm_responses: List[Dict]) -> List[Dict]:
    """
    Analyze LLM responses to identify common patterns.
    Groups similar inputs by offer_type and generates patterns.
    """
    if not llm_responses:
        return []
    
    # Group by offer_type with full result info
    by_type = defaultdict(list)
    for resp in llm_responses:
        if resp.get("output") and resp.get("output") not in ("error", "empty", "unspecified"):
            offer_type = resp["output"]
            input_text = resp.get("input", "").strip()
            full_result = resp.get("full_result", {})
            
            if input_text and len(input_text) > 5:
                by_type[offer_type].append({
                    "text": input_text,
                    "value": full_result.get("value"),
                    "unit": full_result.get("unit")
                })
    
    learned_rules = []
    
    for offer_type, examples in by_type.items():
        if len(examples) < 2:  # Need at least 2 examples to learn a pattern
            continue
        
        # Use LLM to generate pattern from all examples
        example_texts = [e["text"] for e in examples[:10]]  # Use up to 10 examples
        
        # Determine unit from examples
        unit = None
        example_units = [e.get("unit") for e in examples if e.get("unit")]
        if example_units:
            unit = Counter(example_units).most_common(1)[0][0]
        
        # Generate pattern using LLM (intelligent pattern extraction)
        pattern = extract_pattern_with_llm(example_texts, offer_type, unit)
        
        # Fallback to simple pattern if LLM fails
        if not pattern and example_texts:
            pattern = extract_pattern_from_text_simple(example_texts[0], offer_type, unit)
        
        if pattern:
            most_common_pattern = pattern
            
            # Determine unit from examples
            unit = None
            example_texts = [e["text"] for e in examples]
            example_units = [e.get("unit") for e in examples if e.get("unit")]
            
            if example_units:
                unit = Counter(example_units).most_common(1)[0][0]
            elif "%" in " ".join(example_texts):
                unit = "%"
            elif "$" in " ".join(example_texts) or "£" in " ".join(example_texts):
                unit = "USD" if "$" in " ".join(example_texts) else "GBP"
            
            learned_rules.append({
                "offer_type": offer_type,
                "pattern": pattern,
                "examples": example_texts[:5],  # Keep first 5 examples
                "unit": unit,
                "count": len(examples),
                "priority": 90,  # Lower than built-in rules
                "created": datetime.now().isoformat(),
                "generated_by": "llm"
            })
    
    return learned_rules

def apply_learned_rules(raw: str) -> Optional[Dict[str, Any]]:
    """
    Apply learned rules to a text.
    Returns parsed result if matched, None otherwise.
    """
    rules = load_learned_rules()
    if not rules:
        return None
    
    for rule in rules:
        pattern = rule.get("pattern")
        if not pattern:
            continue
        
        try:
            if re.search(pattern, raw, re.IGNORECASE):
                offer_type = rule["offer_type"]
                unit = rule.get("unit")
                
                result = {
                    "offer_type": offer_type,
                    "value": None,
                    "unit": unit,
                    "modifier": [],
                    "stackable": "unknown",
                    "language": "en",
                    "raw": raw
                }
                
                # Try to extract value
                if unit == "%":
                    pm = re.search(r"(?i)(\d+(?:\.\d+)?)\s*%", raw)
                    if pm:
                        result["value"] = float(pm.group(1))
                elif unit in ("USD", "GBP"):
                    mm = re.search(r"(?i)[$£]\s*(\d+(?:\.\d+)?)", raw)
                    if mm:
                        result["value"] = float(mm.group(1))
                        result["currency"] = unit
                
                return result
        except Exception:
            continue
    
    return None

