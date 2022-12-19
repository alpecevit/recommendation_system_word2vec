from nltk.tokenize import word_tokenize


# function for data grouping
def data_grouping_function(df_name, df_grouped_column, categorical_column):
    df = df_name.groupby(str(df_grouped_column)).agg({str(categorical_column): lambda x: ' '.join(x)}).reset_index()
    df['products'] = [word_tokenize(text) for text in df[str(categorical_column)].to_list()]
    return df