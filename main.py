import os
import json
import re
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core import Settings

# --- LLM and Embedding Setup ---
llm_model = "meta/llama-3.1-405b-instruct"
nvllm = NVIDIA(
    model=llm_model,
    api_key="Mentioned_Api_Key"
)

Settings.llm = nvllm
Settings.context_window = 4096
Settings.num_output = 1100

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def extract_entities(text):
    prompt = f"""
    Extract the following medical entities from the patient forum text below:
    - Drugs (medications mentioned)
    - Adverse Drug Events (ADEs)
    - Symptoms/Diseases

    Return your answer as a JSON object with the format:
    {{
        "drugs": [list of drug names],
        "ades": [list of adverse drug events],
        "symptoms": [list of symptoms or diseases]
    }}

    Patient forum text:
    \"\"\"{text}\"\"\"
    """
    response = Settings.llm.complete(prompt, max_tokens=Settings.num_output)
    if not isinstance(response, str):
        response = str(response) if response is not None else ""
    try:
        entities = json.loads(response)
    except Exception:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            entities = json.loads(match.group())
        else:
            raise ValueError(f"LLM did not return valid JSON. Response was: {response}")
    return entities

# --- Main Extraction Loop ---
sample_dir = "data/cadec/sample_text"
entities_dict = {}

for filename in os.listdir(sample_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(sample_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            sample_text = f.read()
        try:
            entities = extract_entities(sample_text)
            entities_dict[filename] = entities
            print(f"Extracted entities for {filename}:")
            print(json.dumps(entities, indent=2))
        except Exception as e:
            print(f"Error extracting entities for {filename}: {e}")

# --- Save all extracted entities to a single JSON file ---
output_path = "extracted_entities.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(entities_dict, f, indent=2, ensure_ascii=False)
print(f"\nSaved all extracted entities to {output_path}")

# --- Optionally, save each file's entities separately ---
output_dir = "extracted_json"
os.makedirs(output_dir, exist_ok=True)
for filename, entities in entities_dict.items():
    json_filename = filename + ".json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2, ensure_ascii=False)
    print(f"Saved extracted entities for {filename} to {json_path}")

# --- Validation/Verification Functions ---
def load_ann_file(ann_dir, ann_filename):
    ann_path = os.path.join(ann_dir, ann_filename)
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(ann_path, "r", encoding="latin-1") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(ann_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

def load_all_ann_files(ann_dir):
    ann_files_content = {}
    for filename in os.listdir(ann_dir):
        if filename.endswith('.ann'):
            ann_files_content[filename] = load_ann_file(ann_dir, filename)
    return ann_files_content

def parse_ann_file_content(ann_content):
    drugs, ades, symptoms = set(), set(), set()
    drug_keywords = ['arthrotec', 'lipitor', 'aspirin', 'ibuprofen']
    ade_keywords = ['drowsy', 'blurred vision', 'malaise', 'agony', 'severe pain', 'weird']
    symptom_keywords = ['arthritis', 'gastric', 'pain', 'problems']
    for line in ann_content.splitlines():
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue
        entity_label = parts[2].strip().lower()
        if any(kw in entity_label for kw in drug_keywords):
            drugs.add(entity_label)
        elif any(kw in entity_label for kw in ade_keywords):
            ades.add(entity_label)
        elif any(kw in entity_label for kw in symptom_keywords):
            symptoms.add(entity_label)
    return {
        'drugs': list(drugs),
        'ades': list(ades),
        'symptoms': list(symptoms)
    }

def compute_similarity(entity, gt_terms):
    if not gt_terms:
        return 0.0, ""
    corpus = [entity] + gt_terms
    vectorizer = TfidfVectorizer().fit_transform(corpus)
    vectors = vectorizer.toarray()
    sim_scores = cosine_similarity([vectors[0]], vectors[1:])
    max_sim = float(np.max(sim_scores))
    best_match = gt_terms[int(np.argmax(sim_scores))]
    return max_sim, best_match

def validate_entities(entities, ground_truth, threshold=0.7):
    validation_results = {}
    for category in entities:
        validation_results[category] = []
        gt_terms = [term.lower() for term in ground_truth.get(category, [])]
        for entity in entities[category]:
            sim_score, best_match = compute_similarity(entity.lower(), gt_terms)
            is_valid = sim_score >= threshold
            validation_results[category].append({
                'entity': entity,
                'is_valid': bool(is_valid),
                'similarity_score': float(sim_score),
                'best_match': best_match
            })
    return validation_results

# --- Verification System (placeholders for format/completeness/semantic) ---
def verify_format(entry: dict) -> bool:
    return True

def verify_completeness(entry: dict) -> bool:
    return True

def verify_semantic(entry: dict) -> bool:
    return True

def verify_entry(entry: dict) -> bool:
    return all([verify_format(entry), verify_completeness(entry), verify_semantic(entry)])

def create_correction_prompt(entry: dict, doc_name: str) -> str:
    return (
        f"The annotation for `{doc_name}` failed validation.\n"
        "Please fix the JSON so it meets schema, completeness, and semantic requirements. "
        "Respond with only the corrected JSON.\n\n"
        f"Input JSON:\n{json.dumps(entry, indent=2)}"
    )

def validate_and_correct(
    all_ann_contents: dict,
    llm,
    max_retries: int = 3
) -> dict:
    corrected = {}
    llm_name = getattr(llm, 'model', getattr(llm, '__name__', str(llm)))
    print(f"Using LLM: {llm_name}")
    for doc, entry in all_ann_contents.items():
        print(f"\nValidating {doc}...")
        logger.info(f"Validating annotation for {doc}")

        if verify_entry(entry):
            print(f"✅ {doc} is valid; no correction needed.")
            corrected[doc] = entry
            continue

        for attempt in range(1, max_retries + 1):
            print(f"Attempt {attempt} to correct {doc}...")
            logger.info(f"Correction attempt {attempt} for {doc}")
            prompt = create_correction_prompt(entry, doc)
            try:
                resp = llm(prompt)
                print(f"💬 Raw LLM response: {resp}")
                new_entry = json.loads(resp)
            except Exception as e:
                print(f"⚠️ LLM error on attempt {attempt} for {doc}: {e}")
                logger.warning(f"LLM error on {doc}, attempt {attempt}: {e}")
                continue

            if verify_entry(new_entry):
                print(f"✅ Correction succeeded for {doc} on attempt {attempt}")
                logger.info(f"Successful correction for {doc}")
                corrected[doc] = new_entry
                break
            else:
                print(f"❌ Correction did not pass verification on attempt {attempt}")
                logger.warning(f"Correction failed verification for {doc}, attempt {attempt}")
        else:
            print(f"⚠️ Max retries reached for {doc}; using original.")
            logger.error(f"Max retries reached for {doc}")
            corrected[doc] = entry

    return corrected

# --- Example Usage for Validation ---
if __name__ == "__main__":
    # Example: validate extracted entities against .ann files
    ann_dir = "data/cadec/sample_scr"
    all_ann_contents = load_all_ann_files(ann_dir)
    for filename, ann_content in all_ann_contents.items():
        ground_truth = parse_ann_file_content(ann_content)       
        print(f"Results for {filename}:")
        print("\n")
