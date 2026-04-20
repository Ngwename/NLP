
# NLP Text Processing and Named Entity Recognition Demo

## Table of Contents

* [Overview](#overview)
* [What the Code Does](#what-the-code-does)
  * [1) Tokenization, Stopword Removal, POS Tagging, and Lemmatization](#1-tokenization-stopword-removal-pos-tagging-and-lemmatization)
  * [2) Named Entity Recognition and Pronoun Ambiguity Detection](#2-named-entity-recognition-and-pronoun-ambiguity-detection)
* [Requirements](#requirements)
* [How to Run](#how-to-run)
* [Results Interpretation](#results-interpretation)
* [Outputs](#outputs)

---

## Overview

This project demonstrates two foundational **Natural Language Processing (NLP)** workflows in Python using **NLTK** and **spaCy**:

1. **Text preprocessing with NLTK**
   - Tokenization
   - Stopword removal
   - Part-of-speech (POS) tagging
   - Lemmatization
   - Filtering to keep only **nouns** and **verbs**

2. **Named Entity Recognition (NER) with spaCy**
   - Identification of entities such as people, organizations, and locations
   - Detection of possible **pronoun ambiguity** in text

These examples are designed for use in a **Jupyter Notebook** and provide a simple introduction to preprocessing and entity extraction tasks commonly used in NLP pipelines.

---

## What the Code Does

### 1) Tokenization, Stopword Removal, POS Tagging, and Lemmatization

This section uses **NLTK** to preprocess the following input text:

> "John enjoys playing football while Mary loves reading books in the library."

The workflow performs these steps:

#### a. Tokenization
The text is split into individual tokens using `word_tokenize()`.

Example output:

```python
['John', 'enjoys', 'playing', 'football', 'while', 'Mary', 'loves', 'reading', 'books', 'in', 'the', 'library', '.']
````

#### b. Stopword Removal

Common English stopwords such as **while**, **in**, and **the** are removed.
Non-alphabetic tokens such as punctuation are also excluded.

Example output:

```python
['John', 'enjoys', 'playing', 'football', 'Mary', 'loves', 'reading', 'books', 'library']
```

#### c. POS Tagging

Each remaining token is assigned a part-of-speech tag using `nltk.pos_tag()`.

Example output:

```python
[('John', 'NNP'), ('enjoys', 'VBZ'), ('playing', 'VBG'), ('football', 'NN'),
 ('Mary', 'NNP'), ('loves', 'VBZ'), ('reading', 'VBG'), ('books', 'NNS'),
 ('library', 'NN')]
```

#### d. Lemmatization

Each word is converted to its **lemma**, which is its dictionary base form.
For example:

* `enjoys` → `enjoy`
* `playing` → `play`
* `books` → `book`

A helper function maps NLTK POS tags to WordNet POS tags so lemmatization is more accurate.

#### e. Keep Only Nouns and Verbs

Only tokens tagged as **nouns** or **verbs** are retained.

Final output format:

```python
[('John', 'NNP', 'john'),
 ('enjoys', 'VBZ', 'enjoy'),
 ('playing', 'VBG', 'play'),
 ('football', 'NN', 'football'),
 ('Mary', 'NNP', 'mary'),
 ('loves', 'VBZ', 'love'),
 ('reading', 'VBG', 'read'),
 ('books', 'NNS', 'book'),
 ('library', 'NN', 'library')]
```

**Purpose:** demonstrate a standard NLP preprocessing pipeline that prepares text for downstream tasks such as classification, information extraction, or topic analysis.

---

### 2) Named Entity Recognition and Pronoun Ambiguity Detection

This section uses **spaCy** to analyze the following input text:

> "Chris met Alex at Apple headquarters in California. He told him about the new iPhone launch."

The workflow performs these tasks:

#### a. Named Entity Recognition (NER)

The code identifies named entities and their labels using spaCy’s pretrained English model `en_core_web_sm`.

Expected entities include:

* `Chris` → `PERSON`
* `Alex` → `PERSON`
* `Apple` → `ORG`
* `California` → `GPE`

Example output:

```python
Named Entities:
Text: Chris, Label: PERSON
Text: Alex, Label: PERSON
Text: Apple, Label: ORG
Text: California, Label: GPE
```

#### b. Pronoun Ambiguity Detection

The code checks whether the text contains any of the pronouns:

```python
{"he", "she", "they"}
```

If one of these pronouns is found, the program prints:

```python
Warning: Possible pronoun ambiguity detected!
```

This is useful because pronouns such as **he** may refer to more than one previously mentioned person, making the sentence ambiguous.

In the sample sentence, **“He told him”** could refer to either Chris or Alex, so the warning is triggered.

**Purpose:** demonstrate how NER can identify important entities in text while a simple rule-based check can flag passages that may require clarification.

---

## Requirements

Install the required libraries in Jupyter or in your terminal.

### For the NLTK script

```bash
pip install nltk
```

Then download the required NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
```

### For the spaCy script

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

---

## How to Run

### In Jupyter Notebook

1. Open a new notebook.
2. Paste the NLTK code into one cell and run it.
3. Paste the spaCy code into another cell and run it.
4. If you get an error such as:

```python
OSError: [E050] Can't find model 'en_core_web_sm'
```

install the spaCy model using:

```python
!python -m spacy download en_core_web_sm
```

### Suggested Execution Order

1. Install dependencies
2. Download language resources/models
3. Run the NLTK preprocessing script
4. Run the spaCy NER script

---

## Results Interpretation

### Text Preprocessing with NLTK

The first script shows how raw text can be transformed into a cleaner and more structured representation.

* **Tokenization** breaks the sentence into manageable units.
* **Stopword removal** eliminates words that often add little semantic value.
* **POS tagging** identifies each token’s grammatical role.
* **Lemmatization** reduces words to their base forms.
* **Filtering nouns and verbs** keeps the most content-rich terms.

This process is useful for reducing noise in text and improving the quality of features used in NLP tasks.

### NER and Ambiguity Detection with spaCy

The second script identifies real-world entities such as people, organizations, and locations.

In the example sentence:

* `Chris` and `Alex` are recognized as people
* `Apple` is recognized as an organization
* `California` is recognized as a geopolitical entity

The ambiguity warning is also meaningful because the phrase **“He told him”** does not clearly specify which person is speaking and which person is receiving the information.

This combination of NER and ambiguity detection can be helpful in:

* information extraction
* document review
* chatbot input validation
* preprocessing for summarization or relation extraction

---

## Outputs

### Script 1: NLTK Preprocessing

The script prints:

* tokenized text
* filtered tokens after stopword removal
* POS tags
* final list of words showing:

  * original token
  * POS tag
  * lemma

### Script 2: spaCy NER

The script prints:

* all named entities and their labels
* a warning message if ambiguous pronouns are detected

---

## Example Summary

This project provides a simple but practical demonstration of two key NLP tasks:

* **Preprocessing text for analysis**
* **Extracting entities and detecting possible ambiguity**

Together, these scripts show how Python NLP libraries can be used to clean text, identify linguistic structure, extract important entities, and flag potentially unclear references in natural language.



