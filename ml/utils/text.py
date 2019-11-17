"""Provides utility methods for text pre-processing."""

from ml import Context


def nltk_stop_words(language='english'):
    """Return a stopword list in the specified language."""
    import nltk
    from nltk.corpus import stopwords
    nltk.data.path = (Context.artifacts.root,)
    nltk.download('stopwords', download_dir=Context.artifacts.root)
    return stopwords.words(language)
