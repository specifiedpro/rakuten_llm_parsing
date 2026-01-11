# Getting Started Guide

Welcome! This guide will help you get started with the Rakuten Offer Parser Pipeline, even if you've never used it before.

## What is This Project?

This project is a **promotional offer parser** that takes text descriptions of sales and discounts (like "Up to 70% Off + Extra 15% Off") and converts them into structured, machine-readable data.

**Example:**
- **Input:** "Up to 70% Off + Extra 15% Off on orders $150+"
- **Output:** Structured JSON with discount percentages, modifiers, conditions, etc.

The system uses a **hybrid approach**:
- **Fast regex rules** handle most common patterns (95%+ of cases)
- **AI (Google Gemini)** handles edge cases and ambiguous text

---

## Prerequisites

Before you begin, make sure you have:

1. **Python 3.11 or higher** installed
   - Check your version: `python3 --version` or `python --version`
   - Download from [python.org](https://www.python.org/downloads/) if needed

2. **A Google Gemini API key**
   - Get one from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - You'll need a Google account
   - The free tier is usually sufficient for testing

3. **Basic command line knowledge**
   - You should know how to open a terminal/command prompt
   - Know how to navigate directories with `cd`

---

## Step 1: Install Dependencies

1. **Open a terminal** and navigate to the project directory:
   ```bash
   cd /path/to/rakuten
   ```
   (Replace `/path/to/rakuten` with your actual project path)

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **On macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```
   - **On Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   
   You should see `(.venv)` at the start of your command prompt.

4. **Install required packages**:
   ```bash
   pip install -r requirements
   ```

   This will install:
   - pandas (data processing)
   - pydantic (data validation)
   - google-generativeai (AI API)
   - python-dotenv (environment variables)
   - pyyaml (configuration files)

---

## Step 2: Configure Your API Key

1. **Create a `.env` file** in the project root directory:
   ```bash
   touch .env
   ```
   (On Windows, you can create it manually or use `type nul > .env`)

2. **Open the `.env` file** in a text editor and add:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```
   
   Replace `your_actual_api_key_here` with your actual Gemini API key from Google AI Studio.

3. **Save the file**

⚠️ **Important:** Never commit your `.env` file to version control! It contains sensitive information.

---

## Step 3: Prepare Your Input File

The parser expects a CSV file with a column named `promotion_level` containing the offer text.

1. **Check if you have an input file:**
   - The default input file is `offers_output_test.csv`
   - You can see this in `hybrid_parse_offer.py` line 614

2. **Your CSV should look like this:**
   ```csv
   brand_name,promotion_level,use_condition,is_coach_brand
   Coach,Up to 70% Off + Extra 15% Off,on orders $150+,TRUE
   Coach UK,3% Cash Back,NA,TRUE
   Coach CA,低至3折,NA,TRUE
   ```

3. **If you need to use a different file:**
   - Edit `hybrid_parse_offer.py`
   - Find the line: `in_path = "offers_output_test.csv"`
   - Change it to your file name

---

## Step 4: Run the Parser

1. **Make sure your virtual environment is activated** (you should see `(.venv)` in your prompt)

2. **Run the parser:**
   ```bash
   python hybrid_parse_offer.py
   ```

   This uses **async mode** (faster, default).

3. **For debugging, you can use sequential mode:**
   ```bash
   python hybrid_parse_offer.py --sequential
   ```

4. **What to expect:**
   - You'll see progress messages like "📊 Processing X rows..."
   - Processing time will be displayed
   - You'll see which rows used the LLM (AI)
   - The system may learn new patterns automatically

---

## Step 5: Check Your Output

After running, you'll get **two output files**:

### 1. `offers_output_enriched.csv` (Wide Format)
- Original CSV with an added `promotion_parsed` column
- Contains JSON arrays of parsed offers
- Good for viewing alongside original data

**Example:**
```csv
row_id,brand_name,promotion_level,promotion_parsed
0,Coach,"Up to 70% Off + Extra 15% Off","[{""offer_type"":""discount"",""value"":70.0,...}]"
```

### 2. `offers_parsed_exploded.csv` (Long Format)
- Each offer as a separate row
- Flattened structure, easier to analyze
- Good for filtering, grouping, and analysis

**Example:**
```csv
row_id,offer_type,value,unit,modifier,currency,threshold_text,stackable,language,raw
0,discount,70.0,%,["up_to"],,,unknown,en,"Up to 70% Off"
0,discount,15.0,%,["extra"],,,true,en,"Extra 15% Off"
```

---

## Understanding the Output Fields

| Field | Description | Example |
|-------|-------------|---------|
| `offer_type` | Type of offer | `discount`, `cashback`, `shipping`, `gift` |
| `value` | Numeric value | `70.0`, `3.0` |
| `unit` | Unit of value | `%`, `USD`, `GBP` |
| `modifier` | Qualifiers | `["up_to"]`, `["extra"]` |
| `value_low` | Minimum value (for ranges) | `10.0` |
| `value_high` | Maximum value (for ranges) | `70.0` |
| `currency` | Currency code | `USD`, `GBP` |
| `threshold_text` | Condition text | `"on orders $150+"` |
| `stackable` | Can combine with other offers? | `true`, `false`, `unknown` |
| `language` | Detected language | `en`, `zh` |
| `raw` | Original text | `"Up to 70% Off"` |

---

## Common Issues and Solutions

### Issue: "ModuleNotFoundError" or "No module named 'pandas'"
**Solution:** Make sure you:
1. Activated your virtual environment (`source .venv/bin/activate`)
2. Installed requirements (`pip install -r requirements`)

### Issue: "GOOGLE_API_KEY not found" or API errors
**Solution:** 
1. Check that your `.env` file exists in the project root
2. Verify the file contains: `GOOGLE_API_KEY=your_key_here`
3. Make sure there are no extra spaces or quotes around the key
4. Restart your terminal after creating `.env`

### Issue: "The input CSV does not contain a 'promotion_level' column"
**Solution:**
1. Check your CSV file has a column named exactly `promotion_level`
2. Or edit `hybrid_parse_offer.py` to use the correct column name

### Issue: Processing is very slow
**Solution:**
- Make sure you're using async mode (default): `python hybrid_parse_offer.py`
- Don't use `--sequential` unless debugging
- Large files will take time; ~6 minutes for 1000 rows is normal

### Issue: Many "unspecified" offers in output
**Solution:**
- This is normal for generic promotional text like "Special Offers"
- The system only uses AI for structured offers (with numbers/symbols)
- Very generic text will remain "unspecified"

---

## What Happens Behind the Scenes?

1. **Input Processing:** Reads your CSV file
2. **Splitting:** Separates combined offers (e.g., "A + B" becomes two offers)
3. **Rule Matching:** Tries regex patterns first (very fast)
4. **AI Fallback:** For ambiguous cases, calls Google Gemini API
5. **Merging:** Combines rule results with AI results intelligently
6. **Learning:** Automatically learns new patterns from AI responses
7. **Output:** Writes two CSV files

---

## Advanced Features

### Auto-Learning System

The system automatically learns from AI responses and creates new rules:
- After each run, it analyzes patterns
- Generates regex rules for common patterns
- Next run uses these rules instead of calling AI
- Gets faster and cheaper over time!

**View learned rules:**
```bash
cat rules/learned_rules.json
```

**Reset learned rules:**
```bash
rm rules/learned_rules.json
```

### Custom Rules

You can add your own parsing rules:
1. Create a Python file in `rules/custom/`
2. Write a function that returns a dict with `offer_type`
3. Register it in `rules/registry.yaml`
4. See `README.md` for detailed instructions

---

## Next Steps

1. **Try with your own data:**
   - Prepare a CSV with `promotion_level` column
   - Update the input file path in `hybrid_parse_offer.py`
   - Run the parser

2. **Analyze the results:**
   - Open the output CSVs in Excel or a data analysis tool
   - Check which offers were parsed successfully
   - Look for patterns in "unspecified" offers

3. **Read the full documentation:**
   - See `README.md` for detailed technical documentation
   - Check `AUTO_LEARNING.md` for learning system details

4. **Customize for your needs:**
   - Add custom rules for your specific offer formats
   - Adjust the LLM prompt if needed
   - Modify output formats

---

## Getting Help

If you encounter issues:

1. **Check the error message** - it usually tells you what's wrong
2. **Review this guide** - common issues are covered above
3. **Check the README.md** - more detailed technical information
4. **Look at the code comments** - the code is well-documented

---

## Quick Reference

**Essential commands:**
```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements

# Run parser (async, fast)
python hybrid_parse_offer.py

# Run parser (sequential, for debugging)
python hybrid_parse_offer.py --sequential

# View learned rules
cat rules/learned_rules.json
```

**File locations:**
- Input: `offers_output_test.csv` (or your file)
- Output: `offers_output_enriched.csv` and `offers_parsed_exploded.csv`
- Config: `.env` (create this)
- Rules: `rules/learned_rules.json` (auto-generated)

---

## Summary

You're all set! The basic workflow is:

1. ✅ Install dependencies
2. ✅ Set up API key in `.env`
3. ✅ Prepare your CSV file
4. ✅ Run `python hybrid_parse_offer.py`
5. ✅ Check the output files

**Happy parsing! 🚀**

