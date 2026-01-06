# YIELD: A Large-Scale Dataset for Information Elicitation Dialogue

For the regular and experimental YIELD datasets, see the `datasets` folder in the root of this repository.  

## Getting Started

Create a file `config/config.yaml` with data and models paths as shown below.

```         
paths:
  proj_store: "/data/yield_dataset" # Actual path to data
  models: "/data/models" # Actual path to models
```


## Downloading models.

-   Llama models can be downloaded from `https://huggingface.co/meta-llama`.
-   DeepSeek models can be downloaded from `https://huggingface.co/deepseek-ai`.

## ORL and SFT

The main training scripts are `./elicitation_agents/experiments/agent_llama.py` (ORL)and `./elicitation_agents/experiments/supervised_finetuning.py` (SFT). Run these file from the root folder.

Example usage:

```         
accelerate launch \
    --config_file config/accelerate_config.yaml \
    ./sequential_ieas/experiments/agent_llama.py \
    --model_choice meta-llama/Llama-3.1-8B-Instruct \
    --dataset_choice yield-v1-experimental \
    --nametag seq-stdt
```


## Obtaining Conformity utterances

There script for obtaining utterances is available at `./elicitation_agents/evaluation/generate_dialogues.py`.
