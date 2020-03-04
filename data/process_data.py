import logging
import pandas as pd
from sqlalchemy import create_engine
import sys
import os

logger = logging.getLogger('disaster_response_etl')
logger.setLevel(logging.INFO)
console_handle = logging.StreamHandler(sys.stdout)
console_handle.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s -%(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handle.setFormatter(formatter)
logger.addHandler(console_handle)


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(left=messages, right=categories, on=['id'])
    logger.info('The data has loaded successfully!')
    return df


def clean_data(df: pd.DataFrame, col: str = 'categories') -> pd.DataFrame:
    categories = df[col].str.split(';', expand=True)
    row = categories.loc[0, :]
    category_colnames = [i.split('-')[0] for i in row.values]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])
    df.drop(col, axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    logger.info('The data has been cleaned!')
    return df


def save_data(df: pd.DataFrame, database_filepath: str):
    data_path = os.path.join(f'sqlite:///{database_filepath}')
    engine = create_engine(data_path)
    return df.to_sql('messages_disaster', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)
        print('Cleaning data...')
        df = clean_data(df)

        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
