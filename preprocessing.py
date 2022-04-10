"""This module is used to perform preprocessing of data."""
from nltk import word_tokenize
from nltk import pos_tag
from nltk import download
from nltk.data import find
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import ssl


class PreprocessingData:
    """Preprocess data.
    Reference: https://machinelearningknowledge.ai/
    11-techniques-of-text-preprocessing-using-nltk-in-python/
    """

    def __init__(self) -> None:
        """Constructor
        """
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        try:
            find('tokenizers/stopwords.zip')
        except LookupError:
            download('stopwords', quiet=True)
        try:
            find('tokenizers/punkt.zip')
        except LookupError:
            download('punkt', quiet=True)
        try:
            find('tokenizers/averaged_perceptron_tagger.zip')
        except LookupError:
            download('averaged_perceptron_tagger', quiet=True)
        try:
            find('tokenizers/omw-1.4.zip')
        except LookupError:
            download('omw-1.4', quiet=True)
        try:
            find('tokenizers/wordnet.zip')
        except LookupError:
            download('wordnet', quiet=True)
        self.en_stopwords = stopwords.words('english')
        self.tokenizer = RegexpTokenizer(r"\w+")
        self.lemmatizer = WordNetLemmatizer()
        self.porter = PorterStemmer()

    def perform_lower_casing(self, series: object) -> object:
        """Perform lower casing on series of text.
        Need this: helpful in text featurization techniques
        like term frequency, TFIDF since it prevents duplication
        of same words having different casing.
        """
        return series.str.lower()

    def _remove_spaces(self, text):
        return " ".join(text.split())

    def remove_extra_whitespaces(self, series: object) -> object:
        """Remove extra whitespaces in series.
        Need this: extra spaces
        increase the text size and
        not add any value to the data.
        """
        return series.apply(self._remove_spaces)

    def perform_tokenization(self, series: object) -> object:
        """Perform tokenization on series of text.
        Need this: prerequisites for many NLP operations.
        """
        return series.apply(lambda X: word_tokenize(X))

    def _remove_stopwords(self, text):
        result = []
        for token in text:
            if token not in self.en_stopwords:
                result.append(token)
        return result

    def remove_stopwords(self, series: object) -> object:
        """Remove stopwords from series of text.
        Need this: appear so frequently in the text that they
        may distort many NLP operations without adding much
        valuable information
        """
        return series.apply(self._remove_stopwords)

    def _remove_punct(self, text):
        list_text = self.tokenizer.tokenize(' '.join(text))
        return list_text

    def remove_punctuations(self, series: object) -> object:
        """Remove punctuations from series of text.
        Need this: punctuations does not add any value to the information.
        This is a text standardization process that will help to treat words like
        'some.', 'some,', and 'some' in the same way.
        """
        return series.apply(self._remove_punct)

    def _lemmatize_text(self, text):
        result = []
        for token, tag in pos_tag(text):
            pos = tag[0].lower()
            if pos not in ['a', 'r', 'n', 'v']:
                pos = 'n'
            result.append(self.lemmatizer.lemmatize(token, pos))
        return result

    def perform_lemmatize(self, series: object) -> object:
        """Perform lemmatize on series of text.
        Need this: create better features for machine learning and NLP models.
        Why use POS: some words are treated as a noun in the given sentence
        rather than a verb. To overcome come this, we use POS (Part of Speech) tags.
        """
        return series.apply(self._lemmatize_text)

    def _stemming(self, text):
        result = []
        for word in text:
            result.append(self.porter.stem(word))
        return result

    def perform_stemming(self, series: object) -> object:
        """Perform stemming on series of text.
        Need this: reduces the words to their root forms
        but unlike lemmatization, the stem itself may not
        a valid word in the Language.
        """
        return series.apply(self._stemming)
