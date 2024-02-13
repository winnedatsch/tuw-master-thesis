# TU Wien Master Thesis - GS-VQA
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/winnedatsch/tuw-master-thesis-evaluation)

## About
This repository hosts the code for Jan Hadl's Master Thesis at TU Wien: GS-VQA, a zero-shot visual questions answering (VQA) pipeline that uses vision-language models (VLMs) for visual perception and answer-set programming (ASP) for symbolic reasoning. 

**Title**: GS-VQA: Zero-Shot Neural-Symbolic Visual Question Answering with Vision-Language Models

**Advisor**: O.Univ.Prof. Dipl.-Ing. Dr.techn. Thomas Eiter

**Assistance**: Projektass. Dipl.-Ing. Dr.techn. Johannes Oetsch, Bakk.techn. and Projektass. Nelson Nicolas Higuera Ruiz, Magíster en Ciencias

**Programme**: MSc Logic and Computation

## Branches 
- `main`: Contains the main pipeline implementation and the results for the evaluation of the core pipeline and the pipeline with fine-tuned CLIP (auxiliary research question 2, or ARQ2) on GQA's `testdev` set
- `feature/spatial-relation-handling`: Contains modifications of the pipeline for the explicit computation of spatial relations between objects and an evaluation run on the `testdev` set to answer the first component of ARQ3
- `feature/llm-relation-scoring`: Contains modifications of the pipeline for the integration of LLMs to judge the plausibility of object relations and an evaluation run on the `testdev` set to answer the second component of ARQ3
- `feature/theory-cleanup`: Contains some clean-up of the ASP theory to make it more consistent in formatting and naming

## Reproducability
To reproduce the main evaluation run of the thesis, a Kaggle Notebook is provided that contains (with some path corrections) the contents of `notebooks/evaluation.ipynb` running on GQA's `testdev_balanced_questions.json` dataset using the CLIP (ViT-B/32) and OWL-ViT (ViT-L/14) models: https://www.kaggle.com/code/winnedatsch/tuw-master-thesis-evaluation.
