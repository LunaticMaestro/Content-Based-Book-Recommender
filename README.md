---
title: Book Recommender
emoji: ⚡
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.6.0
app_file: app.py
pinned: false
short_description: A content based book recommender.
---

# Content-Based-Book-Recommender
A HyDE based approach for building recommendation engine.

Try it out: https://huggingface.co/spaces/LunaticMaestro/book-recommender

![image](.resources/preview.png)

## Foreword

- All images are my actual work please source powerpoint of them in `.resources` folder of this repo.

- Code is documentation is as per [Google's Python Style Guide](https://google.github.io/styleguide/pyguide.html).

- ALL files Paths are at set as CONST in beginning of each script, to make it easier while using the paths while inferencing & evaluation; hence not passing as CLI arguments.

- Seed value for code reproducability is set at as CONST as well.

- prefix `z_` in filenames is just to avoid confusion (to human) of which is prebuilt module and which is custom during import.

## Table of Content

> 

- [Running Inference Locally](#libraries-execution)
- [10,000 feet Approach overview](#approach)
- Pipeline walkthrough in detail

  *For each part of pipeline there is separate script which needs to be executed, mentioned in respective section along with output screenshots.*
  - [Training](#training-steps)
    - [Step 1: Data Clean](#step-1-data-clean)
    - [Step 2: Generate vectors of the books summaries](#step-2-generate-vectors-of-the-books-summaries)
    - [Step 3: Fine-tune GPT-2 to Hallucinate but with some bounds.](#step-3-fine-tune-gpt-2-to-hallucinate-but-with-some-bounds)

  - [Evaluation](#evaluation)
  - Inference
## Running Inference Locally

### Memory Requirements

The code need <2Gb RAM to use both the following. Just CPU works fine for inferencing.

  - https://huggingface.co/openai-community/gpt2 ~500 mb
  - https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 <500 mb


### Libraries 
I used google colab with following libraries extra.

```SH
pip install sentence-transformers datasets
```

### Running 

#### Local System 

```SH 
python app.py
```
access at http://localhost:7860/ 

#### Goolge Colab 

Modify app.py edit line 93 to `demo.launch(share=True)` then run following in cell.

```
!python app.py
```

## Approach

![image](.resources/approach.png)

References:
- This is the core idea: https://arxiv.org/abs/2212.10496
- Another work based on same, https://github.com/aws-samples/content-based-item-recommender
- For future, a very complex work https://github.com/HKUDS/LLMRec

## Training Steps

### Step 1: Data Clean

What is taken care
  - unwanted column removal (the first column of index)
  - missing values removal (drop rows)
  - duplicate rows removal.
  
What is not taken care
  - stopword removal, stemming/lemmatization or special character removal 
  
  **because approach is to use the casual language modelling (later steps) hence makes no sense to rip apart the word meaning**


### Observations from `z_cleand_data.ipynb`

- Same title corresponds to different categories

  ![image](.resources/clean_1.png)

- Total 1230 unique titles.

  ![image](.resources/clean_2.png)

**Action**: We are not going to remove them rows that shows same titles (& summaries) with different categories but rather create a separate file for unique titles.

**RUN**: 

```SH
python z_clean_data.py
```

![image](.resources/clean_3.png)


Output: `clean_books_summary.csv`, `unique_titles_books_summary.csv`


### Step 2: Generate vectors of the books summaries.

**WHAT & WHY** 


Here, I am going to use pretrained sentence encoder that will help get the meaning of the sentence. We perform this over `unique_titles_books_summary.csv` dataset

Caching because the semantic meaning of the summaries (for books to output) are not changed during entire runtime.

![image](.resources/generate_emb.png)


**RUN**:

Use command
```SH
python z_embedding.py
```

Just using CPU should take <1 min

![image](.resources/generate_emb2.png)


Output: `app_cache/summary_vectors.npy`

### Step 3: Fine-tune GPT-2 to Hallucinate but with some bounds.

**What & Why**

Hypothetical Document Extraction (HyDE) in nutshell
  - The **Hypothetical** part of HyDE approach is all about generating random summaries,in short hallucinating. **This is why the approach will work for new book titles**
  - The **Document Extraction** (part of HyDE) is about using these hallucinated summaries to do semantic search on database. 


**Why to fine-tune GPT-2**

1. We want it to hallucinate but withing boundaries i.e. speak words/ language that we have in books_summaries.csv NOT VERY DIFFERENT OUT OF WORLD LOGIC.

2. Prompt Tune such that we can get consistent results. (Screenshot from https://huggingface.co/openai-community/gpt2); The screenshot show the model is mildly consistent.

  ![image](.resources/fine-tune.png)
   
Reference: 
- HyDE Approach, Precise Zero-Shot Dense Retrieval without Relevance Labels https://arxiv.org/pdf/2212.10496
- Prompt design and book summary idea I borrowed from https://github.com/pranavpsv/Genre-Based-Story-Generator
  - I didnt not use his model
      - its lacks most of the categories; (our dataset is different)
      - His code base is too much, can edit it but not worth the effort.
- Fine-tuning code instructions are from https://huggingface.co/docs/transformers/en/tasks/language_modeling

**RUN**


If you want to 

  - push to HF. You must supply your token from huggingface, required to push model to HF

    ```SH
    huggingface-cli login
    ```
  
  - Not Push to HF, then in `z_finetune_gpt.py`:

    - set line 59 ` push_to_hub` to `False`
    - comment line 77 `trainer.push_to_hub()`

We are going to use dataset `clean_books_summary.csv` while triggering this training.

```SH
python z_finetune_gpt.py
```

Image below just shows for 2 epochs, but the one push to my HF https://huggingface.co/LunaticMaestro/gpt2-book-summary-generator is trained for 10 epochs that lasts ~30 mins for 10 epochs with T4 GPU **reduing loss to 0.87 ~ (perplexity = 2.38)**

![image](.resources/fine-tune2.png)


The loss you see is cross-entryopy loss; as ref in the [fine-tuning instructions](https://huggingface.co/docs/transformers/en/tasks/language_modeling) : `Transformers models all have a default task-relevant loss function, so you don’t need to specify one `

So all we care is lower the value better is the model trained :) 

We are NOT going to test this unit model on some test dataset as the model is already proven (its GPT-2 duh!!).
But **we are going to evaluate our HyDE approach end-2-end next to ensure sanity of the approach** that will inherently prove the goodness of this model.

## Evaluation

Before discussing evaluation metric let me walk you through two important pieces recommendation generation and similarity matching;

### Recommendation Generation

The generation is handled by functions in script `z_hypothetical_summary.py`. Under the hood following happens

![image](.resources/eval1.png)

**Function Preview** I did the minimal post processing to chop of the `prompt` from the generated summaries before returning the result.

![image](.resources/eval2.png)

### Similarity Matching 

![image](.resources/eval3.png)

![image](.resources/eval4.png)

**Function Preview** Because there are 1230 unique titles so we get the averaged similarity vector of same size.

![image](.resources/eval5.png)

### Evaluation Metric & Result

So for given input title we can get rank (by desc order cosine similarity) of the store title. To evaluate we the entire approach we are going to use a modified version **Mean Reciprocal Rank (MRR)**.

![image](.resources/eval6.png)

Test Plan:
  - Take random 30 samples and compute the mean of their reciprocal ranks. 
  - If we want that our known book titles be in top 5 results then MRR >= 1/5 = 0.2

**RUN**

```SH 
python z_evaluate.py
```

![image](.resources/eval7.png)

The values of TOP_P and TOP_K (i.e. token sampling for our generator model) are sent as `CONST` in the `z_evaluate.py`; The current set of values are borrowed from the work: https://www.kaggle.com/code/tuckerarrants/text-generation-with-huggingface-gpt2#Top-K-and-Top-P-Sampling

MRR = 0.311 implies that there's a good change that the target book will be in rank (1/.311) ~ 3 (third rank) **i.e. within top 5 recommendations**

> TODO: A sampling study can be done to better make this conclusion.

## Inference

`app.py` is written so that it can best work with gradio interface in the HuggingFace, althought you can try it out locally as well :)

```SH
python app.py
```

1. I rewrote the snippets from `z_evaluate.py` to `app.py` with minor changes to expriment with view. 
2. DONT set `debug=True` for gradio in HF space, else it doesn't start.
3. HF space work differently for retaining models across module scipts; local running (tried in colab space) works  faster. You will see lot of my commits in HF Space to work around this problem.





