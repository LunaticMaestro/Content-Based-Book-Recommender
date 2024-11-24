# THIS file is meant to be used once hence not having functions just sequential code
# Fine-tuning code instructions are from https://huggingface.co/docs/transformers/en/tasks/language_modeling


import pandas as pd
from transformers import AutoTokenizer, set_seed
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from z_utils import get_dataframe

# CONST 
INP_DATASET_CSV = "clean_books_summary.csv" 
BASE_CASUAL_MODEL = "openai-community/gpt2"
TRAINED_MODEL_OUTPUT_DIR = "gpt2-book-summary-generator" # same name for HF Hub
set_seed(42)
EPOCHS = 2 # 10
LR = 2e-5

# Load dataset
books: pd.DataFrame = get_dataframe(INP_DATASET_CSV)

# Create HF dataset, easier to perform preprocessing at scale
dataset_books = Dataset.from_pandas(books, split="train")

# Loading Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_CASUAL_MODEL) 

# Data Preprocessing
def preprocess_function(book):
    '''Funtion to convert dataset to in prompt form
    '''
    # Its Multiline, so DONT put tabs in this editor view otherwise it will get inside string
    text = f'''Genre: {book['categories']}
Book Title: {book['book_name']}
Description: {book['book_name']} {book['summaries']}
'''
    return tokenizer(text)

# Apply Preprocessing
tokenized_dataset_books = dataset_books.map(
    preprocess_function,
    # batched=True,
    num_proc=4,
    remove_columns=dataset_books.column_names,
)

# Data Collator, req for Casual LM
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# Load Casual LM
model = AutoModelForCausalLM.from_pretrained(BASE_CASUAL_MODEL)
training_args = TrainingArguments(
    output_dir=TRAINED_MODEL_OUTPUT_DIR,
    eval_strategy="no",
    learning_rate=LR,
    weight_decay=0.01,
    push_to_hub=True,
    num_train_epochs=EPOCHS,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_books,
    # eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Commit model files to HF
trainer.push_to_hub()