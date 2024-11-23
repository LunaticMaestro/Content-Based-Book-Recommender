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

## Libraries installed separately

I used google colab with following libraries extra. 

```SH
pip install -U sentence-transformers datasets
```

## Training Steps

**ALL files Paths are at set as CONST in beginning of each script, to make it easier while using the paths while inferencing; hence not passing as CLI arguments**

### Step 1: Data Clean

I am going to do basic steps like unwanted column removal (the first column of index), missing values removal (drop rows), duplicate rows removal. Output Screenshot attached.

I am NOT doing any text pre-processing steps like stopword removal, stemming/lemmatization or special character removal because my approach is to use the casual language modelling (later steps) hence makes no sense to rip apart the word meaning via these word-based techniques.

A little tinker in around with the dataset found that some titles can belong to multiple categories. (*this code I ran separately, is not part of any script*)

![image](https://github.com/user-attachments/assets/cdf9141e-21f9-481a-8b09-913a0006db87)

A descriptive analysis shows that there are just 1230 unique titles. (*this code I ran separately, is not part of any script*)

![image](https://github.com/user-attachments/assets/072b4ed7-7a4d-48b2-a93c-7b08fc5bee45)

We are not going to remove them rows that shows same titles (& summaries) with different categories but rather create a separate file for unique titles.

```SH
python z_clean_data.py
```

![image](https://github.com/user-attachments/assets/a466c20b-60ed-47ac-8bfc-e0a38ccdb88d)


Output: `clean_books_summary.csv`, `unique_titles_books_summary.csv`


### Step 2: Generate vectors of the books summaries.

Here, I am going to use pretrained sentence encoder that will help get the meaning of the sentence. As the semantic meaning of the summaries themselves are not changed.

We perform this over `unique_titles_books_summary.csv` dataset

![image](https://github.com/user-attachments/assets/21d2d92b-0ad5-4686-8e38-c47df10893f8)

Use command
```SH
python z_embedding.py
```

Just using CPU should take <1 min

![image](https://github.com/user-attachments/assets/5765d586-cc50-4adf-b714-5e371f757f38)

Output: `app_cache/summary_vectors.npy`

### Step 3: Fine-tune GPT-2 to Hallucinate but with some bounds.

Lets address the **Hypothetical** part of HyDE approach. Its all about generating random summaries,in short hallucinating. While the **Document Extraction** (part of HyDE) is about using these hallucinated summaries to do semantic search on database. 

Two very important reasons why to fine-tune GPT-2 
1. We want it to hallucinate but withing boundaries i.e. speak words/ language that we have in books_summaries.csv NOT VERY DIFFERENT OUT OF WORLD LOGIC.

2. Prompt Tune such that we can get consistent results. (Screenshot from https://huggingface.co/openai-community/gpt2); The screenshot show the model is mildly consistent.

  ![image](https://github.com/user-attachments/assets/1b974da8-799b-48b8-8df7-be17a612f666)
  
  > we are going to use ``clean_books_summary.csv` dataset in this training to align with the prompt of ingesting different genre.
   
Reference: 
- HyDE Approach, Precise Zero-Shot Dense Retrieval without Relevance Labels https://arxiv.org/pdf/2212.10496
- Prompt design and book summary idea I borrowed from https://github.com/pranavpsv/Genre-Based-Story-Generator
  - I didnt not use his model
      - its lacks most of the categories; (our dataset is different)
      - His code base is too much, can edit it but not worth the effort.
- Fine-tuning code instructions are from https://huggingface.co/docs/transformers/en/tasks/language_modeling

Command

You must supply your token from huggingface, required to push model to HF

```SH
huggingface-cli login
```

We are going to use dataset `clean_books_summary.csv` while triggering this training.

```SH
python z_finetune_gpt.py
```
(Training lasts ~30 mins for 10 epochs with T4 GPU)

![image](https://github.com/user-attachments/assets/46253d48-903a-4977-b3f5-39ea1e6a6fd6)


The loss you see is cross-entryopy loss; as ref in the fine-tuning instructions (see above reference) states : `Transformers models all have a default task-relevant loss function, so you don’t need to specify one `

![image](https://github.com/user-attachments/assets/13e9b868-6352-490c-9803-c5e49f8e8ae8)

So all we care is lower the value better is the model trained :) 

We are NOT going to test this unit model for some test dataset as the model is already proven (its GPT-2 duh!!).
But **we are going to evaluate our HyDE approach end-2-end next to ensure sanity of the approach**.

## Evaluation

Before discussing evaluation metric let me walk you through two important pieces recommendation generation and similarity matching;

### Recommendation Generation

The generation is handled by script `z_hypothetical_summary.py`. Under the hood following happens

![image](https://github.com/user-attachments/assets/ee174c38-a1f3-438a-afb8-be2888c590da)

Code Preview. I did the minimal post processing to chop of the `prompt` from the generated summaries before returning the result.

![image](https://github.com/user-attachments/assets/132e84a7-cb4f-49d2-8457-ff473224bad6)

### Similarity Matching 

![image](https://github.com/user-attachments/assets/229ce58b-77cb-40b7-b033-c353ee41b0a6)

![image](https://github.com/user-attachments/assets/58613cd7-0b73-4042-b98d-e6cdf2184c32)

Because there are 1230 unique titles so we get the averaged similarity vector of same size.

![image](https://github.com/user-attachments/assets/cc7b2164-a437-4517-8edb-cc0573c8a5e6)

### Evaluation Metric

So for given input title we can get rank (by desc order cosine similarity) of the store title. To evaluate we the entire approach we are going to use a modified version **Mean Reciprocal Rank (MRR)**.

![image](https://github.com/user-attachments/assets/0cb8fc2a-8834-4cda-95d2-52a02ac9c11d)

We are going to do this for random 30 samples and compute the mean of their reciprocal ranks. Ideally all the title should be ranked 1 and their MRR should be equal to 1. Closer to 1 is good.

![image](https://github.com/user-attachments/assets/d2c77d47-9244-474a-a850-d31fb914c9ca)

The values of TOP_P and TOP_K (i.e. token sampling for our generator model) are sent as `CONST` in the `z_evaluate.py`; The current set of values of this are borrowed from the work: https://www.kaggle.com/code/tuckerarrants/text-generation-with-huggingface-gpt2#Top-K-and-Top-P-Sampling

MRR = 0.311 implies that there's a good change that the target book will be in rank (1/.311) ~ 3 (third rank) **i.e. within top 5 recommendations**







