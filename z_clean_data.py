from z_utils import get_dataframe

# Const
ORIGNAL_DF = "books_summary.csv"
CLEAN_DF = "cleaned_"+ORIGNAL_DF 
CLEAN_DF_UNIQUE_TITLES = "unique_titles_"+ORIGNAL_DF

# Load dataset
books_df = get_dataframe(ORIGNAL_DF) 

# Original stats
print(f"Original Shape: {books_df.shape}")

# Drop Unknown columns 
req_columns = ['book_name', 'summaries', 'categories']
books_df = books_df[req_columns] # another way could be .drop(...)

# Check for nulls
print(f"\n\nNulls Count=== \n{books_df.isna().sum()}")
# removing nulls rowsise cuz their other attirbutes dont contribute 
books_df.dropna(axis=0, inplace=True) 


# Check & remove duplciates
print(f"\n\nDuplicate Records: {books_df.duplicated().sum()}")
books_df.drop_duplicates(inplace=True) 


# Final stats
print(f"\n\nCleaned Shape: {books_df.shape}")

# Saving these cleaned DF
print("Storing cleaned as (this includes same titles with diff cats: "+CLEAN_DF)
books_df.to_csv(CLEAN_DF, index=False)

# ==== NOW to store the unique titles  ====
books_df = books_df[["book_name", "summaries"]]
books_df.drop_duplicates(inplace=True)
print(f"\n\nDF w/ unique titles Shape: {books_df.shape}")
# Saving these cleaned DF
print("Storing dataset w/ unqiue titles & summaries only "+CLEAN_DF_UNIQUE_TITLES)
books_df.to_csv(CLEAN_DF_UNIQUE_TITLES, index=False)
