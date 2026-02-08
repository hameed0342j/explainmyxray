"""
Script to convert medical jargon to patient-friendly text.
Uses Gemini API (or can be adapted for OpenAI/local models).
"""
import os
import pandas as pd
from tqdm import tqdm
from typing import Optional

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


SIMPLIFY_PROMPT = """You are a medical translator. Convert this radiology report into simple, 
patient-friendly language that a non-medical person can understand.

Rules:
1. Use everyday words (e.g., "collapsed lung" instead of "pneumothorax")
2. Explain what the finding means for the patient
3. Be reassuring but accurate
4. Keep it concise (2-3 sentences max)
5. Do NOT provide medical advice or diagnosis

Medical Report:
{report}

Patient-Friendly Explanation:"""


def simplify_with_gemini(report: str, api_key: Optional[str] = None) -> str:
    """
    Use Gemini API to convert medical jargon to simple text.
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
    
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set. Set it as environment variable.")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = SIMPLIFY_PROMPT.format(report=report)
    response = model.generate_content(prompt)
    
    return response.text.strip()


def simplify_basic(report: str) -> str:
    """
    Basic rule-based simplification (fallback when API not available).
    """
    # Common medical term translations
    translations = {
        "pneumothorax": "collapsed lung",
        "cardiomegaly": "enlarged heart",
        "pleural effusion": "fluid around the lungs",
        "consolidation": "area of infection in the lung",
        "infiltrates": "signs of infection or inflammation",
        "atelectasis": "partially collapsed lung tissue",
        "nodule": "small spot or growth",
        "opacity": "cloudy area",
        "costophrenic": "lower corner of the lung",
        "hilar": "central part of the lung",
        "bilateral": "both sides",
        "unilateral": "one side",
        "normal": "appears healthy with no concerning findings",
        "unremarkable": "looks normal",
        "no acute": "nothing urgent",
        "fibrosis": "scarring",
        "emphysema": "damage to air sacs in lungs",
        "COPD": "chronic lung disease",
        "aortic elongation": "stretching of the main blood vessel",
        "scoliosis": "curved spine",
        "fracture": "broken bone",
    }
    
    result = report.lower()
    for medical, simple in translations.items():
        result = result.replace(medical.lower(), simple)
    
    return result.capitalize()


def process_csv(
    input_csv: str,
    output_csv: str,
    report_column: str = "Report",
    use_api: bool = True,
    api_key: Optional[str] = None,
):
    """
    Process a CSV file and add simplified reports.
    
    Args:
        input_csv: Path to input CSV
        output_csv: Path to save output CSV
        report_column: Column containing medical reports
        use_api: Whether to use Gemini API (True) or basic rules (False)
        api_key: Gemini API key (optional, uses env var if not provided)
    """
    df = pd.read_csv(input_csv)
    
    simplified = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Simplifying reports"):
        report = str(row[report_column]) if pd.notna(row[report_column]) else ""
        
        if not report or report.strip() == "":
            simplified.append("")
            continue
        
        try:
            if use_api and GEMINI_AVAILABLE:
                simple = simplify_with_gemini(report, api_key)
            else:
                simple = simplify_basic(report)
            simplified.append(simple)
        except Exception as e:
            print(f"Error at row {idx}: {e}")
            simplified.append(simplify_basic(report))
    
    df["SimplifiedReport"] = simplified
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved to {output_csv}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplify medical reports")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--column", default="Report", help="Report column name")
    parser.add_argument("--no-api", action="store_true", help="Use basic rules instead of API")
    args = parser.parse_args()
    
    process_csv(
        input_csv=args.input,
        output_csv=args.output,
        report_column=args.column,
        use_api=not args.no_api,
    )
