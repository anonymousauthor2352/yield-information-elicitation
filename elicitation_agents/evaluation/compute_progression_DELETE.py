"""
Usage:
    python ./sequential_ieas/evaluation/compute_progression.py
"""

# ---------------- Imports ----------------
import os

from datetime import datetime
from transformers import AutoTokenizer

import torch
import yaml

from sentence_transformers import CrossEncoder

from elicitation.metrics.progression import progression
from elicitation.metrics.utils import load_dialogues



# ---------------- Args ----------------
# Paths

cross_encoder_model_choice_name = "cross-encoder/stsb-roberta-large"
tokenizer_model_choice_name = "meta-llama/Llama-3.2-3B-Instruct"

dataset_choice_name = "evaluation/generated-utterances-dialogue/20251218T2008-20251216t1321-llama-3.1-8b-instruct-seq-std-m4trained-m3generated/generated"





# Constants
PROGRESSION_K = 5
PROGRESSION_GAMMA = 0.5
#CONVERSATIONAL_CONTROL_K = 2 
#CONVERSATIONAL_CONTROL_GAMMA = 0.9



# %%
# ---------------- Config ----------------
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

proj_store = config["paths"]["proj_store"]

data_path = os.path.join(config["paths"]["proj_store"], "data")

INPUT_FILE = os.path.join(proj_store, dataset_choice_name)


output_path = os.path.join(proj_store, "evaluation", "progression", f"{dataset_choice_name}")
os.makedirs(output_path, exist_ok=True)
output_file = os.path.join(output_path, f"progression.csv")



models_folderpath = config["paths"]["models"]

cross_encoder_model_choice = os.path.join(models_folderpath, cross_encoder_model_choice_name)
tokenizer_model_choice = os.path.join(models_folderpath, tokenizer_model_choice_name)



# Load llama tokenizer
tokenizer_model = AutoTokenizer.from_pretrained(tokenizer_model_choice, trust_remote_code=True)

# Load sentence cross encoder model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
cross_encoder = CrossEncoder(cross_encoder_model_choice, device=device)


all_dialogues = list(load_dialogues(INPUT_FILE))

print("Loaded", len(all_dialogues), "dialogues")



# ---------------- Main ----------------
def main():
        
    print(f"\nGenerating from {INPUT_FILE}")
    progression_df = progression(
        dialogues=all_dialogues, 
        cross_encoder=cross_encoder, 
        k=PROGRESSION_K, 
        gamma=PROGRESSION_GAMMA, 
        group_by="domain", 
        sort_by="domain"
    )
    print(progression_df)
    
    progression_df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")



# ------------------------
# EXECUTION
# ------------------------

if __name__ == "__main__":
    
    main()

