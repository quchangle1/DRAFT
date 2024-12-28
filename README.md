# DRAFT
The implementation for the paper: From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions.

## How to run the code
1. Download ToolBench dataset from the [Google Drive](https://drive.google.com/drive/folders/1yBUQ732mPu-KclJnuQELEhtKakdXFc3J).
2. Run DRAFT to get revised tool documentation:
	> python DRAFT.py
3. Run ToolBench_DFSDT to examine the effectiveness of DRAFT:
	> python ToolBench_DFSDT -model_name gpt-4o-2024-08-06 -data_type G3 -method DRAFT

## Environment

Our experimental environment is shown below:

```
openai version: 0.28.0
numpy version: 1.26.4
pandas version: 2.2.2
torch version: 2.3.1
```
