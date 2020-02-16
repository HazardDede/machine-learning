from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

from ml.utils.text import nltk_stop_words

NGRAM_SPACE = ((1, 1), (1, 2), (1, 3))
USE_IDF_SPACE = (True, False)
SVD_COMPONENT_SPACE = (10, 20, 50)


def feature_space(language, enable_count_vector=True, enable_tfidf=True):
    space = []

    stop_words = nltk_stop_words(language)

    if enable_count_vector:
        # Only count vector
        cvect = {
            'p1': (CountVectorizer(stop_words=stop_words, max_df=0.7, min_df=0.01),),
            'p1__ngram_range': NGRAM_SPACE,
            'p2': (None,),
            'p3': (TruncatedSVD(n_components=30, random_state=42),)
        }
        space.append(cvect)

    if enable_tfidf:
        # TF-IDF Vectorizer preprocessing
        tfidfv = {
            'p1': (None,),
            'p2': (TfidfVectorizer(stop_words=stop_words, max_df=0.7, min_df=0.01),),
            'p3': (TruncatedSVD(n_components=30, random_state=42),),
            'p2__ngram_range': NGRAM_SPACE,
            'p2__use_idf': USE_IDF_SPACE,
            'p2__sublinear_tf': (True, False)
        }
        space.append(tfidfv)

    if not space:
        raise RuntimeError("You have to enable at least one feature engineering technique.")

    return space


def feature_steps():
    return [
        ('p1', None),
        ('p2', None),
        ('p3', None),
    ]
