# hybrid_parse_offers.py
# Reads offers_output_clean.csv, parses promotion_level using hybrid rules+Gemini,
# writes offers_output_enriched.csv (wide) and offers_parsed_exploded.csv (long).
import os
import time, pathlib
from time import sleep
import asyncio
from typing import Awaitable
from rules.loader import load_registry, load_custom_rule_functions
from rules.learned_rules import (
    load_learned_rules, save_learned_rules, apply_learned_rules,
    analyze_llm_patterns
)

GAP_LOG_PATH = os.getenv("GAP_LOG_PATH", "data/triage/parsing_gaps.jsonl")
SHADOW_RULES = os.getenv("SHADOW_RULES", "false").lower() == "true"

_registry = load_registry()
_custom_rules = load_custom_rule_functions(_registry)  # list of (priority, fn)
import re
import json
import hashlib
from typing import Dict, Any, List, Optional

import pandas as pd
from pydantic import BaseModel, ValidationError, conlist
from dotenv import load_dotenv

# ---------- Env / Gemini ----------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except Exception:
    _GEMINI_AVAILABLE = False

def apply_custom_rules(raw: str) -> Optional[Dict[str, Any]]:
    for _, fn in _custom_rules:
        try:
            out = fn(raw)
            if out and out.get("offer_type") and out.get("offer_type") != "unspecified":
                return out
        except Exception:
            continue
    return None

# ---------- Parsing utilities (rules) ----------
CURRENCY_MAP = {'$': 'USD', '£': 'GBP'}
PERCENT = re.compile(r'(?i)(^|[^A-Z])(\d+(?:\.\d+)?)\s*%')
MONEY = re.compile(r'(?i)([$£])\s*(\d+(?:\.\d+)?)')
MILES = re.compile(r'(?i)\b(\d+(?:\.\d+)?)\s*miles/\$')
UP_TO = re.compile(r'(?i)\bup\s*to\b')
EXTRA = re.compile(r'(?i)\bextra\b')
SUPER = re.compile(r'(?i)\bsuper\b')
MAX = re.compile(r'(?i)\bmax\b')
NOW = re.compile(r'(?i)\bnow\b')
FROM = re.compile(r'(?i)\b(starting\s+at|from)\b')
NO_CASHBACK = re.compile(r'(?i)\b(no|0\s*%?)\s*cash\s*back\b')
COUPONS_ONLY = re.compile(r'(?i)\bcoupons?\s+only\b')
CASHBACK_WORD = re.compile(r'(?i)cash\s*back|cashback|rewards?')
DISCOUNT_WORD = re.compile(r'(?i)\boff|discount')
SHIPPING_FREE = re.compile(r'(?i)\bfree\b.*\bshipping\b')
RETURNS_FREE = re.compile(r'(?i)\bfree\b.*\breturns?\b')
REPAIRS_FREE = re.compile(r'(?i)\bfree\b.*\brepairs?\b')
GIFT_FREE = re.compile(r'(?i)\bfree\b.*\b(gift|bag|blanket|necklace|earring|jewellery box|card case|wristlet)\b')
GIFT_ANY = re.compile(r'(?i)\bfree\b')
CONTEST = re.compile(r'(?i)\b(win|chance\s+to\s+win)\b.*\b(gift\s*card|nintendo|switch|prize)\b')
VIP_PERKS = re.compile(r'(?i)\b(vip|birthday|early\s*access|insider|perks?)\b')
UNSPECIFIED = re.compile(r'(?i)\bunspecified\b|\bna\b|^-$')
RANGE_THRESHOLD = re.compile(r'(?i)(?:orders?|purchase|order|of|on)\s*(?:over|of)?\s*([$£]\s*\d+\+?)')

# 中文規則
ZH_DISCOUNT_LOW_TO = re.compile(r'低至(\d)折')        # 低至3折 -> 70% off
ZH_DISCOUNT_EXTRA = re.compile(r'額外(\d)折')         # 額外7折 -> Extra 30% off
ZH_DISCOUNT_ZHE = re.compile(r'(\d)折')               # 8折 -> 20% off
ZH_CASHBACK_MAX = re.compile(r'最高\s*(\d+(?:\.\d+)?)\s*%?\s*返利')
ZH_REBATE = re.compile(r'(\d+(?:\.\d+)?)\s*%?\s*返利')

def discount_from_zhe(zhe: int) -> float:
    return round((10 - zhe) * 10.0, 2)

def detect_language(s: str) -> str:
    return 'zh' if re.search(r'[\u4e00-\u9fff]', s) else 'en'

def parse_offer(raw: str) -> Dict[str, Any]:
    s = (raw or "").strip()
    
    # Try learned rules first (before built-in rules)
    learned_result = apply_learned_rules(raw)
    if learned_result and learned_result.get("offer_type") != "unspecified":
        return learned_result
    
    result = {
        'offer_type': 'unspecified',
        'value': None,
        'unit': None,
        'modifier': [],
        'value_low': None,
        'value_high': None,
        'currency': None,
        'threshold_text': None,
        'notes': None,
        'stackable': 'unknown',
        'language': detect_language(s),
        'raw': raw
    }

    if not s:
        result['notes'] = ''
        return result

    if EXTRA.search(s): result['modifier'].append('extra'); result['stackable'] = 'true'
    if UP_TO.search(s): result['modifier'].append('up_to')
    if SUPER.search(s): result['modifier'].append('super')
    if MAX.search(s):   result['modifier'].append('max')

    th = RANGE_THRESHOLD.search(s)
    if th: result['threshold_text'] = th.group(0)

    if NO_CASHBACK.search(s):
        result['offer_type'] = 'no_cashback'
        return result

    m = MILES.search(s)
    if m:
        result.update(dict(offer_type='miles', value=float(m.group(1)), unit='miles_per_dollar'))
        return result

    m = ZH_DISCOUNT_LOW_TO.search(s)
    if m:
        result['offer_type'] = 'discount'
        result['value'] = discount_from_zhe(int(m.group(1)))
        result['unit'] = '%'
        if 'up_to' not in result['modifier']: result['modifier'].append('up_to')
        result['value_high'] = result['value']
        return result

    m = ZH_DISCOUNT_EXTRA.search(s)
    if m:
        result['offer_type'] = 'discount'
        result['value'] = discount_from_zhe(int(m.group(1)))
        result['unit'] = '%'
        if 'extra' not in result['modifier']: result['modifier'].append('extra')
        result['stackable'] = 'true'
        return result

    m = ZH_DISCOUNT_ZHE.search(s)
    if m:
        result['offer_type'] = 'discount'
        result['value'] = discount_from_zhe(int(m.group(1)))
        result['unit'] = '%'
        return result

    m = ZH_CASHBACK_MAX.search(s)
    if m:
        result['offer_type'] = 'cashback'
        result['value'] = float(m.group(1))
        result['unit'] = '%'
        if 'up_to' not in result['modifier']: result['modifier'].append('up_to')
        result['value_high'] = result['value']
        return result

    m = ZH_REBATE.search(s)
    if m and '返利' in s:
        result['offer_type'] = 'cashback'
        result['value'] = float(m.group(1))
        result['unit'] = '%'
        return result

    if FROM.search(s):
        m = MONEY.search(s)
        if m:
            cur = CURRENCY_MAP.get(m.group(1))
            result.update(dict(offer_type='price_from',
                               value=float(m.group(2)),
                               unit=cur, currency=cur))
            if 'from' not in result['modifier']: result['modifier'].append('from')
            return result
    if NOW.search(s):
        m = MONEY.search(s)
        if m:
            cur = CURRENCY_MAP.get(m.group(1))
            result.update(dict(offer_type='price_now',
                               value=float(m.group(2)),
                               unit=cur, currency=cur))
            if 'now' not in result['modifier']: result['modifier'].append('now')
            return result

    if SHIPPING_FREE.search(s):
        result.update(dict(offer_type='shipping', value=0, unit='USD', notes='free'))
        return result
    if RETURNS_FREE.search(s):
        result.update(dict(offer_type='returns', value=0, unit='USD', notes='free'))
        return result
    if REPAIRS_FREE.search(s):
        result.update(dict(offer_type='repairs', value=0, unit='USD', notes='free'))
        return result
    if GIFT_FREE.search(s) or (GIFT_ANY.search(s) and 'Free ' in s):
        result.update(dict(offer_type='gift', notes=s))
        return result

    if COUPONS_ONLY.search(s):
        result['offer_type'] = 'coupons_only'; return result
    if CONTEST.search(s):
        result['offer_type'] = 'contest'; result['notes'] = s; return result
    if VIP_PERKS.search(s):
        result['offer_type'] = 'perks'; result['notes'] = s; return result
    if UNSPECIFIED.search(s):
        result['offer_type'] = 'unspecified'; result['notes'] = s; return result

    if CASHBACK_WORD.search(s):
        pm = PERCENT.search(s)
        if pm:
            result.update(dict(offer_type='cashback', value=float(pm.group(2)), unit='%'))
            return result
        mm = MONEY.search(s)
        if mm:
            cur = CURRENCY_MAP.get(mm.group(1))
            result.update(dict(offer_type='cashback', value=float(mm.group(2)), unit=cur, currency=cur))
            return result

    if DISCOUNT_WORD.search(s) or re.search(r'(?i)\b(\d+(\.\d+)?)\s*%\s*off\b', s):
        pm = PERCENT.search(s)
        if pm:
            v = float(pm.group(2))
            result.update(dict(offer_type='discount', value=v, unit='%'))
            if 'up_to' in result['modifier']:
                result['value_high'] = v
            return result
        mm = MONEY.search(s)
        if mm:
            cur = CURRENCY_MAP.get(mm.group(1))
            result.update(dict(offer_type='fixed_off', value=float(mm.group(2)), unit=cur, currency=cur))
            return result

    if re.search(r'(?i)\bdonation\b', s):
        mm = MONEY.search(s)
        if mm:
            cur = CURRENCY_MAP.get(mm.group(1))
            result.update(dict(offer_type='rewards', value=float(mm.group(2)), unit='donation', currency=cur))
            return result

    result['notes'] = s
    return result

def split_promotions(promo: str) -> List[str]:
    if promo is None:
        return []
    parts = re.split(r'\s*\+\s*', promo)
    refined = []
    for p in parts:
        sub_parts = re.split(r'\s+and\s+', p, flags=re.IGNORECASE)
        refined.extend([sp.strip() for sp in sub_parts if sp.strip()])
    return refined

# ---------- Pydantic schema ----------
class Offer(BaseModel):
    offer_type: str
    value: Optional[float] = None
    unit: Optional[str] = None
    modifier: List[str]
    value_low: Optional[float] = None
    value_high: Optional[float] = None
    currency: Optional[str] = None
    threshold_text: Optional[str] = None
    stackable: str
    language: str
    raw: str
    notes: Optional[str] = None
    base_offer_link: Optional[int] = None

class LLMResponse(BaseModel):
    offers: conlist(Offer, min_length=1)
    reasoning: Optional[str] = None

# ---------- Cache ----------
_llm_cache: Dict[str, Dict[str,Any]] = {}

def _cache_key(text: str) -> str:
    norm = " ".join((text or "").split()).lower()
    return hashlib.sha1(norm.encode()).hexdigest()

# ---------- Prompt data ----------
SYSTEM_PROMPT = (
    "You are a promotions parser. Convert marketing text into a strict JSON following the schema. "
    "Do not invent numbers. If text implies extra, set stackable=true and link it to the base offer via base_offer_link. "
    "For up to, use value_high; leave value_low null unless present. Split multiple promos into multiple offers."
)

FEWSHOTS = [
    {
        "TEXT":"Up to 70% Off + Extra 15% Off on Orders $150+",
        "LANG":"en",
        "EXPECTED":[
            {"offer_type":"discount","value":70,"unit":"%","modifier":["up_to"],"value_low":None,"value_high":70,"currency":None,"threshold_text":"on Orders $150+","stackable":"unknown","language":"en","raw":"Up to 70% Off","notes":None,"base_offer_link":None},
            {"offer_type":"discount","value":15,"unit":"%","modifier":["extra"],"value_low":None,"value_high":None,"currency":None,"threshold_text":"on Orders $150+","stackable":"true","language":"en","raw":"Extra 15% Off","notes":"Extra applies to base discount","base_offer_link":0}
        ]
    },
    {
        "TEXT":"低至3折",
        "LANG":"zh",
        "EXPECTED":[
            {"offer_type":"discount","value":70,"unit":"%","modifier":["up_to"],"value_low":None,"value_high":70,"currency":None,"threshold_text":None,"stackable":"unknown","language":"zh","raw":"低至3折","notes":None,"base_offer_link":None}
        ]
    },
    {
        "TEXT":"1.5 miles/$",
        "LANG":"en",
        "EXPECTED":[
            {"offer_type":"miles","value":1.5,"unit":"miles_per_dollar","modifier":[],"value_low":None,"value_high":None,"currency":None,"threshold_text":None,"stackable":"unknown","language":"en","raw":"1.5 miles/$","notes":None,"base_offer_link":None}
        ]
    },
    {
        "TEXT":"Free Shipping + Free Returns",
        "LANG":"en",
        "EXPECTED":[
            {"offer_type":"shipping","value":0,"unit":"USD","modifier":[],"value_low":None,"value_high":None,"currency":None,"threshold_text":None,"stackable":"unknown","language":"en","raw":"Free Shipping","notes":"free","base_offer_link":None},
            {"offer_type":"returns","value":0,"unit":"USD","modifier":[],"value_low":None,"value_high":None,"currency":None,"threshold_text":None,"stackable":"unknown","language":"en","raw":"Free Returns","notes":"free","base_offer_link":None}
        ]
    }
]

# ---------- Gemini client ----------
def _gemini_model():
    if not _GEMINI_AVAILABLE or not GOOGLE_API_KEY:
        return None
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={
                "temperature": 0,
                "response_mime_type": "application/json",
            },
        )
        return model
    except Exception:
        return None

def _build_single_prompt(text: str) -> List[Dict[str, str]]:
    parts: List[Dict[str, str]] = []
    parts.append({"role":"user","parts":[SYSTEM_PROMPT]})
    # Few-shots
    for fs in FEWSHOTS:
        parts.append({"role":"user","parts":[f"TEXT: {fs['TEXT']}\nLANG: {fs['LANG']}"]})
        parts.append({"role":"model","parts":[json.dumps({"offers": fs["EXPECTED"]}, ensure_ascii=False)]})
    # Actual input
    lang = 'zh' if re.search(r'[\u4e00-\u9fff]', text or "") else 'en'
    parts.append({"role":"user","parts":[f"TEXT: {text}\nLANG: {lang}"]})
    return parts

_llm_debug = []  # Global debug list

async def llm_parse_single(model, text: str, semaphore: asyncio.Semaphore, idx: int = -1) -> Dict[str,Any]:
    """Parse a single text with rate limiting via semaphore."""
    async with semaphore:
        try:
            content = _build_single_prompt(text)
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, lambda: model.generate_content(content))
            text_result = (resp.text or "").strip()
            if not text_result:
                result = {"offers":[{"offer_type":"unspecified","modifier":[],"stackable":"unknown","language":"en","raw":text}]}
            else:
                parsed = json.loads(text_result)
                if "offers" not in parsed:
                    result = {"offers":[{"offer_type":"unspecified","modifier":[],"stackable":"unknown","language":"en","raw":text}]}
                else:
                    result = parsed
            # Debug logging and learning
            if idx >= 0:
                offer_type = result["offers"][0]["offer_type"] if result.get("offers") else "empty"
                _llm_debug.append({
                    "row": idx,
                    "input": text[:80],
                    "output": offer_type,
                    "raw_output": text_result[:100] if text_result else "empty",
                    "full_result": result["offers"][0] if result.get("offers") else {}
                })
            return result
        except Exception as e:
            result = {"offers":[{"offer_type":"unspecified","modifier":[],"stackable":"unknown","language":"en","raw":text}]}
            if idx >= 0:
                _llm_debug.append({
                    "row": idx,
                    "input": text[:80],
                    "output": "error",
                    "raw_output": str(e)[:100]
                })
            return result

async def llm_parse_offers_async(texts: List[str], idx_map: List[int] = None) -> List[Dict[str,Any]]:
    """
    Call Gemini concurrently for multiple texts.
    Falls back to unspecified when unavailable.
    """
    model = _gemini_model()
    if model is None:
        return [{"offers":[{"offer_type":"unspecified","modifier":[],"stackable":"unknown","language":"en","raw":t}]} for t in texts]
    
    # Limit concurrent requests to avoid rate limits
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent calls
    if idx_map:
        tasks = [llm_parse_single(model, t, semaphore, idx) for t, idx in zip(texts, idx_map)]
    else:
        tasks = [llm_parse_single(model, t, semaphore) for t in texts]
    return await asyncio.gather(*tasks)

def llm_parse_offers(texts: List[str], idx_map: List[int] = None) -> List[Dict[str,Any]]:
    """
    Synchronous wrapper for async LLM parsing.
    Call Gemini per input and return structured JSON per LLMResponse.
    Falls back to unspecified when unavailable.
    """
    try:
        return asyncio.run(llm_parse_offers_async(texts, idx_map))
    except Exception:
        # Fallback to unspecified if async fails
        return [{"offers":[{"offer_type":"unspecified","modifier":[],"stackable":"unknown","language":"en","raw":t}]} for t in texts]

def llm_parse_offers_sequential(texts: List[str], idx_map: List[int] = None) -> List[Dict[str,Any]]:
    """
    Sequential (line-by-line) LLM parsing for comparison.
    """
    global _llm_debug
    model = _gemini_model()
    out = []
    if model is None:
        return [{"offers":[{"offer_type":"unspecified","modifier":[],"stackable":"unknown","language":"en","raw":t}]} for t in texts]
    
    for idx, t in enumerate(texts):
        try:
            content = _build_single_prompt(t)
            resp = model.generate_content(content)
            text_result = (resp.text or "").strip()
            
            if not text_result:
                result = {"offers":[{"offer_type":"unspecified","modifier":[],"stackable":"unknown","language":"en","raw":t}]}
            else:
                parsed = json.loads(text_result)
                result = {"offers":[{"offer_type":"unspecified","modifier":[],"stackable":"unknown","language":"en","raw":t}]} if "offers" not in parsed else parsed
            
            # Debug logging and learning
            if idx_map and idx < len(idx_map):
                offer_type = result["offers"][0]["offer_type"] if result.get("offers") else "empty"
                _llm_debug.append({
                    "row": idx_map[idx],
                    "input": t[:80],
                    "output": offer_type,
                    "raw_output": text_result[:100] if text_result else "empty",
                    "full_result": result["offers"][0] if result.get("offers") else {}
                })
            out.append(result)
        except Exception as e:
            result = {"offers":[{"offer_type":"unspecified","modifier":[],"stackable":"unknown","language":"en","raw":t}]}
            if idx_map and idx < len(idx_map):
                _llm_debug.append({
                    "row": idx_map[idx],
                    "input": t[:80],
                    "output": "error",
                    "raw_output": str(e)[:100]
                })
            out.append(result)
    return out

# ---------- Merge policy ----------
NUMERIC_FIELDS = ["value","value_low","value_high"]
PRIORITY_RULE = set(NUMERIC_FIELDS + ["unit","currency"])

def merge_rule_and_llm(rule_offers: List[Dict[str,Any]],
                       llm_offers: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    if not rule_offers or all(o.get("offer_type")=="unspecified" for o in rule_offers):
        return llm_offers

    merged = []
    used_llm = [False]*len(llm_offers)

    for r in rule_offers:
        best_j = None
        for j, lo in enumerate(llm_offers):
            if used_llm[j]: continue
            if lo.get("offer_type")==r.get("offer_type"):
                best_j = j; break
        if best_j is None:
            merged.append(r); continue
        lo = llm_offers[best_j]; used_llm[best_j]=True
        m = dict(r)
        for k,v in lo.items():
            if k in PRIORITY_RULE:
                if (m.get(k) is None) and (v is not None): m[k]=v
            elif k=="modifier":
                m[k] = sorted(list(set((m.get(k) or []) + (v or []))))
            elif m.get(k) in (None,"",[]):
                m[k]=v
        merged.append(m)

    for j,lo in enumerate(llm_offers):
        if not used_llm[j]:
            merged.append(lo)
    return merged

# ---------- Driver: rule-first with Gemini fallback ----------
def parse_with_llm_fallback(texts: List[str],
                            rule_parser,
                            batch_size=20,
                            use_async=True) -> List[List[Dict[str,Any]]]:
    results: List[Optional[List[Dict[str,Any]]]] = [None]*len(texts)
    to_call, idx_map = [], []

    def needs_llm(o: Dict[str,Any]) -> bool:
        # Only use LLM for genuinely ambiguous cases
        # Skip generic promotional text, empty strings, etc.
        raw_text = o.get("raw", "").strip().lower()
        
        # Skip empty or very short generic text
        if not raw_text or len(raw_text) < 3:
            return False
        if raw_text in ["special offers", "selected offer", "special offer", "offers", ""]:
            return False
            
        # Only use LLM if we have a specific case
        if o.get("offer_type")=="unspecified" and raw_text and len(raw_text) > 10:
            # Check if it looks like a structured offer we might have missed
            if re.search(r'[\d%$£]', raw_text):  # Has numbers/currency/percentage
                return True
        return False
    
    # Debug: collect unspecified rows
    unspecified_debug = []
    
    model = _gemini_model()
    print(f"📊 Processing {len(texts)} rows...")
    
    # Print debug info for unspecified
    if unspecified_debug:
        print(f"\n⚠️  Found {len(unspecified_debug)} unspecified rows that didn't trigger LLM:")
        for idx, text, res in unspecified_debug[:10]:  # Show first 10
            print(f"  Row {idx}: '{text[:60]}...' -> {res[0].get('offer_type')}")
        if len(unspecified_debug) > 10:
            print(f"  ... and {len(unspecified_debug)-10} more")
        print()

    for i, t in enumerate(texts):
        parts = split_promotions(t or "")
        rule_res = [rule_parser(p) for p in parts] or [rule_parser("")]
        call_needed = any(needs_llm(o) for o in rule_res)
        
        # Debug: track unspecified that didn't trigger LLM
        if not call_needed and any(o.get("offer_type")=="unspecified" for o in rule_res):
            if rule_res[0].get("raw") and len(rule_res[0].get("raw", "")) > 3:
                unspecified_debug.append((i, t, rule_res))

        if not call_needed:
            results[i] = rule_res
            continue

        ck = _cache_key(t or "")
        cached = _llm_cache.get(ck)
        if cached:
            try:
                validated = LLMResponse(**cached).dict()
            except ValidationError:
                validated = {"offers":[{"offer_type":"unspecified","modifier":[],"stackable":"unknown","language":"en","raw":t}]}
            results[i] = merge_rule_and_llm(rule_res, validated["offers"])
            continue

        to_call.append(t); idx_map.append((i, rule_res))

    # Process batches (sequential or async)
    for b in range(0, len(to_call), batch_size):
        chunk = to_call[b:b+batch_size]
        batch_indices = [idx for idx, _ in idx_map[b:b+batch_size]]
        
        if use_async:
            llm_raw = llm_parse_offers(chunk, idx_map=batch_indices)
        else:
            llm_raw = llm_parse_offers_sequential(chunk, idx_map=batch_indices)
            
        for j, llm_obj in enumerate(llm_raw):
            i, rule_res = idx_map[b+j]
            original_text = texts[i]
            try:
                validated = LLMResponse(**llm_obj).dict()
            except ValidationError:
                validated = {"offers":[{"offer_type":"unspecified","modifier":[],"stackable":"unknown","language":"en","raw":original_text}]}
            _llm_cache[_cache_key(original_text or "")] = validated
            merged = merge_rule_and_llm(rule_res, validated["offers"])
            for k,o in enumerate(merged):
                bli = o.get("base_offer_link")
                if isinstance(bli,int) and not (0 <= bli < len(merged)):
                    o["base_offer_link"] = None
            results[i] = merged

    # type: ignore
    return results

# ---------- CSV IO ----------
def main():
    import sys
    global _llm_debug
    _llm_debug = []  # Reset debug list
    
    # Check for sequential mode flag
    use_async = '--sequential' not in sys.argv
    
    if not use_async:
        print("🐌 Using SEQUENTIAL (line-by-line) mode")
    else:
        print("⚡ Using ASYNC (concurrent) mode")
    
    in_path = "offers_output_test.csv"  # Testing with test file
    df = pd.read_csv(in_path)
    if 'promotion_level' not in df.columns:
        raise RuntimeError("The input CSV does not contain a 'promotion_level' column.")

    df = df.reset_index().rename(columns={'index':'row_id'})
    texts = [(str(x) if pd.notna(x) else "") for x in df['promotion_level'].tolist()]
    
    import time
    start_time = time.time()
    merged_offers_per_row = parse_with_llm_fallback(texts, parse_offer, batch_size=50, use_async=use_async)
    elapsed_time = time.time() - start_time
    
    print(f"\n⏱️  Processing time: {elapsed_time:.2f} seconds")
    
    # Print LLM debug info and learn patterns
    if _llm_debug:
        print(f"\n🤖 LLM was used for {len(_llm_debug)} rows:")
        for d in _llm_debug[:10]:  # Show first 10
            print(f"  Row {d['row']}: '{d['input']}' -> {d['output']}")
        if len(_llm_debug) > 10:
            print(f"  ... and {len(_llm_debug)-10} more")
        print()
        
        # Learn patterns from LLM responses
        print("🧠 Learning patterns from LLM responses...")
        new_rules = analyze_llm_patterns(_llm_debug)
        if new_rules:
            existing_rules = load_learned_rules()
            # Merge with existing (avoid duplicates)
            existing_patterns = {r.get("pattern") for r in existing_rules}
            for rule in new_rules:
                if rule.get("pattern") not in existing_patterns:
                    existing_rules.append(rule)
            
            save_learned_rules(existing_rules)
            print(f"  ✅ Learned {len(new_rules)} new patterns:")
            for rule in new_rules[:5]:
                print(f"    - {rule['offer_type']}: {rule['pattern'][:50]}... ({rule['count']} examples)")
            if len(new_rules) > 5:
                print(f"    ... and {len(new_rules)-5} more")
            print(f"  💾 Total learned rules: {len(existing_rules)}")
        else:
            print("  ℹ️  No new patterns to learn (need at least 2 examples per pattern)")
        print()

    # wide
    df['promotion_parsed'] = [
        json.dumps(items, ensure_ascii=False) for items in merged_offers_per_row
    ]

    # long
    exploded_rows: List[Dict[str,Any]] = []
    for rid, offers in zip(df['row_id'].tolist(), merged_offers_per_row):
        for item in offers:
            rec = {'row_id': rid}
            rec.update(item)
            exploded_rows.append(rec)
    long_df = pd.DataFrame(exploded_rows)

    # outputs
    enriched_path = "offers_output_enriched.csv"
    exploded_path = "offers_parsed_exploded.csv"
    df.to_csv(enriched_path, index=False)
    long_df.to_csv(exploded_path, index=False)
    print(f"Wrote {enriched_path} and {exploded_path}")

if __name__ == "__main__":
    main()