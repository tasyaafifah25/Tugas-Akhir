# preprocessing.py

import re
import json
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from html import unescape

# -------------------------------------------
# A. Load resource (slang dictionary, stopwords)
# -------------------------------------------
with open('final_slang.txt', 'r', encoding='utf-8') as f:
    slang_dict = json.load(f)

factory = StopWordRemoverFactory()
stopwords = set(factory.get_stop_words())

# Tambahkan stopwords khusus
custom_stopwords = {
    'huhu', 'wkwk', 'dong', 'akwkakakak', 'hehe', 'akxkakskak', 'sikxskxk', 'wkwkwkwk',
    'wkwkwk', 'wkakka', 'wkkw','wkwkw','wkwkwkwk','wkwkwkwkwk','hehehe','heheh','dongg',
    'wkwwkwk','wkwkwkkw','wkwkwkw','awokwkwkwkw','aowkwoowkwwkwkkw','wkwkwwk','wkwkkww',
    'wkwkk','wkwkwkk','hehehaha','ehehehe','mwehehe','hihi','wkwkak','wkwkwkkwk','wkwkwkwkwkwkwk',
    'kwkwkw','wkwkkw','wksokwowkwowk','wkwww','wkkwkw','wlwkwl','akwkw','wkakaka','akowkwokwok',
    'wkakak','wkw','awowkwowok','kwkakakak','awkawkawkkk','wk','wkkakak','nya'
}
stopwords.update(custom_stopwords)

# -------------------------------------------
# B. Definisi fungsi-fungsi bantu
# -------------------------------------------
def REPLACE(text, old, new):
    return text.replace(old, new)

def REMOVE_USERNAME(text):
    return re.sub(r'@\w+', '', text)

def REMOVE_HASHTAG(text):
    return re.sub(r'#\w+', '', text)

def REMOVE_RETWEET(text):
    return re.sub(r'RT\s+', '', text)

def REMOVE_URL(text):
    return re.sub(r'http\S+|www\.\S+', '', text)

def REMOVE_EMOJI(text):
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def REMOVE_NON_ASCII(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def REMOVE_NUMBERS(text):
    return re.sub(r'\d+', '', text)

def REMOVE_EXTRA_SPACES(text):
    return re.sub(r'\s+', ' ', text).strip()

def CASE_FOLDING(text):
    return text.lower()

def REMOVE_PUNCTUATION(text):
    return re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_{|}~]', ' ', text)

def NORMALIZE_WORDS(text):
    words = text.split()
    normalized_words = [slang_dict.get(word.lower(), word) for word in words]
    return ' '.join(normalized_words)

def REMOVE_STOPWORDS(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered_words)

def TOKENIZE_TEXT(text):
    return text.split()

# -------------------------------------------
# FUNGSI: Preprocessing Hanya Sampai Stopword Removal
# -------------------------------------------
def preproc_until_stopwords(text):
    """
    Melakukan semua langkah pembersihan sampai (dan termasuk) stopword removal.
    Mengembalikan string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Beberapa penggantian khusus
    text = REPLACE(text, "\\\\r", " ")
    text = REPLACE(text, "\n", " ")
    text = REPLACE(text, "&amp;", " ")

    # Hapus mention (@username), hashtag (#tag), RT, URL, emoji, dsb.
    text = REMOVE_USERNAME(text)
    text = REMOVE_HASHTAG(text)
    text = REMOVE_RETWEET(text)
    text = REMOVE_URL(text)
    text = REMOVE_EMOJI(text)
    text = REMOVE_NON_ASCII(text)
    text = REMOVE_NUMBERS(text)
    text = REMOVE_EXTRA_SPACES(text)

    # Unescape HTML entities (misal &gt; -> '>')
    text = unescape(text)

    # Case folding (lowercase)
    text = CASE_FOLDING(text)

    # Hapus tanda baca
    text = REMOVE_PUNCTUATION(text)

    # Normalisasi kata slang
    text = NORMALIZE_WORDS(text)

    # Hapus stopwords
    text = REMOVE_STOPWORDS(text)

    return text  # string yang sudah bebas stopword

# -------------------------------------------
# FUNGSI: Tokenisasi Akhir
# -------------------------------------------
def final_tokenize(text):
    """
    Melakukan tokenisasi (atau langkah lanjutan) dari hasil string
    yang sudah bebas stopwords.
    """
    if not text or not isinstance(text, str):
        return []
    return TOKENIZE_TEXT(text)
