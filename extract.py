import os
import ast
from google import genai
import re
import json
import glob

def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def generate_citations(text: str, client, prompt_template: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_template + "\n\n" + text
    )
    return response.text

def clean_ai_output(output: str) -> str:
    for fence in ["```json", "```text", "```"]:
        output = output.replace(fence, "")
    return output.strip()

def parse_ai_output(output: str) -> list[str]:
    output = clean_ai_output(output)


    # Convert to Python list
    titles = ast.literal_eval(output)
    return titles

def classify_and_normalize(title, last_rule=None):
    title_lower = title.lower()
    normalized = None

    # -------- 1. Jurisprudence --------
    if "g.r. no." in title_lower or " v. " in title_lower or " vs. " in title_lower or " vs " in title_lower:
        # Normalize "vs" to "v"
        normalized = re.sub(r'\bvs\.?\s+', 'v. ', title, flags=re.IGNORECASE)
        return "Jurisprudence", normalized, last_rule

    # -------- 2. Constitution --------
    elif "const." in title_lower or "constitution" in title_lower or "article" in title_lower:
        art_match = re.search(r'article\s*(\w+)', title_lower)
        sec_match = re.search(r'section\s*(\d+)', title_lower)
        if art_match:
            art_num = art_match.group(1).upper()
            if sec_match:
                sec_num = sec_match.group(1)
                normalized = f"C{art_num}S{sec_num}"
            else:
                normalized = f"C{art_num}"
        else:
            normalized = title
        return "Constitution", normalized, last_rule

    # -------- 3. Civil Code --------
    cc_match = re.search(r'(civil code).*?(article|art\.)\s*(\d+)', title_lower)
    if cc_match:
        article_num = cc_match.group(3)
        normalized = f"CC{article_num}"
        return "Statute", normalized, last_rule

    # -------- 4. Statutes --------
    # Check for Act (statute)
    act_match = re.search(r'act\s+(\d+)', title_lower)
    if act_match:
        act_num = act_match.group(1)
        normalized = f"A{act_num}"
        return "Statute", normalized, last_rule
    
    if re.search(r'r\.a\.|pd|bp', title_lower):
        statute_match = re.search(r'(r\.a\.|pd|bp)\s*(\d+)', title_lower)
        if statute_match:
            prefix = statute_match.group(1).upper().replace('.', '')
            num = statute_match.group(2)
            normalized = f"{prefix}{num}"
        else:
            normalized = title
        return "Statute", normalized, last_rule

    # -------- 5. Administrative Rules (Rules of Court, IRR, Circulars) --------
    rule_match = re.search(r'rules?\s*(\d+)', title_lower)
    section_match = re.search(r'section\s*(\d+)', title_lower)

    if rule_match:
        rule_num = rule_match.group(1)
        last_rule = rule_num  # update last cited rule
        if section_match:
            sec_num = section_match.group(1)
            normalized = f"R{rule_num}S{sec_num}"
        else:
            normalized = f"R{rule_num}"
        return "Administrative rule", normalized, last_rule
    
    elif section_match and last_rule:
        sec_num = section_match.group(1)
        normalized = f"R{last_rule}S{sec_num}"
        return "Administrative rule", normalized, last_rule

    # -------- 6. Anything else --------
    return "Other", title, last_rule

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment variables.")
    client = genai.Client(api_key=api_key)

    prompt = """
    Extract only the citations that appear verbatim in the text. 
    Do NOT create or invent any citations that are not present. 

    Requirements:

    1. Return only the exact citation titles as they appear in the text.
       For Sections, include only the section number (e.g., "Section 2"), do not include summaries.
    2. Include only citations from these sources: Constitution, Statutes, Jurisprudence, Administrative rules.
    3. Citation types are: Constitution, Statute, Jurisprudence, Administrative rule. Do not include any other types.
    4. Do not include commentary, explanations, or extra text. Only return the list of titles.

    Output format (strictly):
    ["Title1","Title2","Title3",...]

    - Maintain the order in which the citations appear in the text.
    - Do NOT add extra spaces or line breaks within the list.
    - Ensure each title is enclosed in quotes and separated by commas.
    """

    # Iterate through all .txt files in the cases folder
    cases_folder = "cases"
    txt_files = glob.glob(os.path.join(cases_folder, "*.txt"))
    
    # Sort files to process in order (case1.txt, case2.txt, etc.)
    txt_files.sort()
    
    # Store sequences separately for each case (to avoid cross-case transitions)
    all_titles_sequences = []  # List of lists - each case is a separate sequence
    all_types_sequences = []   # List of lists - each case is a separate sequence
    
    for txt_file in txt_files:
        print(f"Processing: {txt_file}")
        text = read_file(txt_file)
        output = generate_citations(text, client, prompt)
        
        # Parse AI output into a list of titles
        titles = parse_ai_output(output)
        
        # Classify and normalize each title for this case
        case_titles = []
        case_types = []
        last_rule = None
        for t in titles:
            classification, normalized, last_rule = classify_and_normalize(t, last_rule)
            # Skip "Other" classifications - don't count them
            if classification != "Other":
                case_titles.append(normalized)
                case_types.append(classification)
        
        # Store this case as a separate sequence
        if case_titles:  # Only add non-empty sequences
            all_titles_sequences.append(case_titles)
            all_types_sequences.append(case_types)
    
    # Print results
    print(f"\nProcessed {len(all_titles_sequences)} cases")
    print("Total sequences:", all_types_sequences)
    
    # Save as list of sequences (nested list structure)
    with open("title_sequence.json", "w") as f:
        json.dump(all_titles_sequences, f)

    with open("types_sequence.json", "w") as f:
        json.dump(all_types_sequences, f)
    
    # Also save flattened version for backward compatibility if needed
    all_titles_flat = [item for seq in all_titles_sequences for item in seq]
    all_types_flat = [item for seq in all_types_sequences for item in seq]
    
    with open("title_sequence_flat.json", "w") as f:
        json.dump(all_titles_flat, f)
    
    with open("types_sequence_flat.json", "w") as f:
        json.dump(all_types_flat, f)    

if __name__ == "__main__":
    main()
