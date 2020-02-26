import re
import sys
import logging
import joblib
import nltk
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score,\
                            recall_score, f1_score
from xgboost import XGBClassifier
import warnings

nltk.download(['punkt', 'wordnet', 'stopwords'])

warnings.filterwarnings(action='ignore')

logger = logging.getLogger('app_disaster_response')
logger.setLevel(logging.INFO)
console_handle = logging.StreamHandler(sys.stdout)
console_handle.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s -%(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handle.setFormatter(formatter)
logger.addHandler(console_handle)


def load_data(database_filepath: str, table_name: str = 'messages_disaster'):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(table_name=table_name, con=engine)

    cols = df.columns[4:]

    X = df['message']
    y = df.loc[:, cols]
    logger.info('The data has been loaded')
    return X, y


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    words = word_tokenize(text=text)
    words = [word for word in words if word not in stopwords.words('english')]
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]
    return words


def build_model():
    gb_pipeline = Pipeline([('tfidf', TfidfVectorizer(min_df=0.1, tokenizer=tokenize)),
                            ('clf', MultiOutputClassifier(XGBClassifier(random_state=42, n_jobs=-1)))])
    params = {
        'tfidf__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__max_depth': [5, 7, 10]
    }

    model = GridSearchCV(estimator=gb_pipeline, param_grid=params, n_jobs=-1, cv=5)
    logger.info('The model has been instantiated')
    return model


def evaluate_model(model, X_test: pd.DataFrame, Y_test: pd.Series):
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': np.round(accuracy_score(y_true=Y_test, y_pred=y_pred), 2),
        'precision': np.round(precision_score(y_true=Y_test, y_pred=y_pred, average='weighted'), 2),
        'recall': np.round(recall_score(y_true=Y_test, y_pred=y_pred, average='weighted'), 2),
        'f1': np.round(f1_score(y_true=Y_test, y_pred=y_pred, average='weighted'), 2),

    }
    for i, col in enumerate(Y_test):
        print(f'The metrics for {col} is: \n')
        print(f'The accuracy is: {metrics["accuracy"]}')
        print(f'The precision is: {metrics["precision"]}')
        print(f'The recall is: {metrics["recall"]}')
        print(f'The F1 score is: {metrics["f1"]}')
    logger.info('The model has been evaluated on unobserved data')


def save_model(model, model_filepath: str):
    joblib.dump(model, model_filepath)
    logger.info('The model has been saved successfully!')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
