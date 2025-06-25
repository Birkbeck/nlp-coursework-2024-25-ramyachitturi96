#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
import pandas as pd
from pathlib import Path
import pickle
from collections import Counter
import math


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    sentence = nltk.sent_tokenize(text)
    Words = nltk.word_tokenize(text)
    words = []
    for word in Words:
        if word.isalpha():
            words.append(word.lower())
    total_sentences = len(sentence)
    total_words = len(words) 
    total_syllables = 0
    for word in words:
        total_syllables += count_syl(word, d)
    fk_grade = (0.39 * ((total_words) / (total_sentences))) + (11.8 * ((total_syllables) / (total_words))) - 15.59
    return fk_grade
    pass


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    syllables_count = 0
    if word in d:
        phonome_list = d[word][0]
        for p in phonome_list:
            if p[-1].isdigit():
                syllables_count +=1
    else: 
        vowels = "aeiouy" #y is considered as vowels in some cases as it gives vowel sound
        prev_char_was_vowel = False 
        for w in word:
            if w in vowels and not prev_char_was_vowel:
                syllables_count += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        if word.endswith("le") and word[-3] in vowels:
            syllables_count -= 1
    return syllables_count
    pass


def read_novels(path=Path.cwd() / "p1-texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    data = []
    for file in path.glob('*.txt'):
        title_name, author, year = file.stem.split("-")
        text = file.read_text(encoding="utf-8")
        title = title_name.replace("_", " ")
        data.append({"text":text, "title":title, "author":author, "year":int(year)})
    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by="year").reset_index(drop=True)
    return df_sorted
    pass


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    #add a columm to dataframe
    df["parsed"] = df["text"].apply(nlp)

    # Serialise the resulting dataframe
    output_path = store_path/out_name
    store_path.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as file:
        pickle.dump(df, file)
    
    #Return dataframe
    return df

    pass


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    final_tokens = []
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        if token.isalnum():
            final_tokens.append(token.lower())
    ttr = len(set(final_tokens)) / len(final_tokens)
    return ttr
    pass


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subject_count = Counter()              #subject count in text
    subject_target_verb_count = Counter()  #subject related to target verb
    verb_count = 0                         #Number of verbs
    total_tokens = len(doc)                # Length of tokens

    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            subject_count[token.lemma_.lower()] += 1
        if token.lemma_.lower() == target_verb.lower() and token.pos_ in ["VERB", "AUX"]:
            verb_count += 1
            for child in token.children:
                if child.dep_ in ('nsubj', 'nsubjpass'):
                    subject_target_verb_count[child.lemma_.lower(), token.text.lower()] += 1


    pmi_value = []
    prob_verb = verb_count / total_tokens

    for (subject, verb), count in subject_target_verb_count.items():
        if subject not in subject_count or subject_count[subject] == 0:
            continue
        prob_subject = subject_count[subject] / total_tokens
        prob_subj_verb = count / total_tokens
        pmi = math.log2(prob_subj_verb / (prob_subject * prob_verb))
        pmi_value.append((subject, verb, pmi, count))

    return sorted(pmi_value, key = lambda x:x[1], reverse=True)[:10]
    pass

def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subjects = Counter()
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ == verb:
            for child in token.children:
                if child.dep_ in ('nsubj','nsubjpass'):
                    subjects[child.text.lower()] += 1
    return subjects.most_common(10)
    
    pass
def common_objects(doc):
    objects = Counter()
    for token in doc:
        if token.dep_ == "dobj" and token.head_pos == "VERB":
            objects[token.text.lower()] += 1
    return objects.most_common(10)


def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    adjectives = Counter()
    for doc in df["parsed"]:
        for token in doc:
            if token.pos_ == "ADJ":
                adjectives[token.lemma_.lower()] += 1
    return adjectives.most_common(10)
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    #nltk.download("cmudict")
    parse(df)
    print(df.head())
    print("TTR")
    print(get_ttrs(df))
    print("\n")
    print("FKS")
    print(get_fks(df))
    print("\n")
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    print("pickle file created")
    print("\n")
    print("adjective_count")
    print(adjective_counts(df))
    print("\n")
    print("subject by verb count")
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")
    print("subject by verb PMI")
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    

