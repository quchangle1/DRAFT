# DRAFT
The implementation for ICLR 2025 Submission: From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions.

## How to run the code
1. Download ToolBench dataset
2. Run DRAFT to get revised tool documentation:
	> python DRAFT.py
3. Run ToolBench_DFSDT to examine the effectiveness of DRAFT:
	> python ToolBench_DFSDT -model_name gpt-4o-2024-08-06 -data_type G3 -method DRAFT

## Environment

Our experimental environment is shown below:

```
numpy version: 1.26.4
pandas version: 2.2.2
torch version: 2.3.1
```