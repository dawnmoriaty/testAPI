import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def combine_datasets(english_path, vietnamese_path):
    df_en = load_data(english_path)
    df_vi = load_data(vietnamese_path)
    df_en['language'] = 'en'
    df_vi['language'] = 'vi'
    return pd.concat([df_en, df_vi], ignore_index=True)
