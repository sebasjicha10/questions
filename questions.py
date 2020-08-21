import nltk
from nltk.corpus import stopwords
import sys
import os
import string
import math


FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as f:

            # Add key - value pairs to files
            content = f.read()
            files[filename] = content

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    words = nltk.word_tokenize(document.lower())

    contents = [
        word for word in 
        words 
        if word not in string.punctuation and word not in stopwords.words('english')
    ]

    return contents


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    # Get all words in corpus
    words = set()

    for filename in documents.values():
        for word in filename:
            words.add(word)

    idfs = dict()

    # Get the idf for each word
    for word in words:


        f = sum(word in documents[filename] for filename in documents)

        idf = math.log(len(documents) / f)
        idfs[word] = idf

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    # Put files in a list
    filenames = [filename for filename in files]

    # Handle sorting
    def handleSort(f):
        score = 0
        
        # Loop through query words and file word and compute score
        for query_word in query:

            if query_word in files[f]:

                tf = files[f].count(query_word)
                idf = idfs[query_word]
                tf_idf = tf * idf
                score += tf_idf

        return score

    # Sort list
    filenames.sort(reverse=True, key=handleSort)
    filenames = filenames[:n]

    return filenames

   
def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    # List of sentences
    sentences_options = [sentence for sentence in sentences]

    # Handle sorting
    def handleSort(f):

        score = 0

        # Sort by idf
        for query_word in query:

            if query_word in sentences[f]:
                idf = idfs[query_word]
                score += idf

        return score

    # Query Termn Density
    def handleDensity(f):

        density = 0
        sentence_len = len(sentences[f])
    
        for query_word in query:

            if query_word in sentences[f]:

                density += 1
 
        return density / sentence_len

    # Sort sentences
    idf_ranked = sorted(sentences_options, reverse=True, key=lambda rank: (handleSort(rank), handleDensity(rank)))
    idf_ranked = idf_ranked[:n]

    return idf_ranked
    

if __name__ == "__main__":
    main()
