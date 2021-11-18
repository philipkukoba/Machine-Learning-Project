from itertools import combinations
from queue import Queue
# from graph import Graph
# from preprocessing import TextProcessor
from gensim.models import KeyedVectors
import re
import string
import unicodedata
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
# from syntactic_unit import SyntacticUnit

class SyntacticUnit(object):
    """
    Wrapper class for words, processed tokens, corresponding part-of-speech tags and scores
    """

    def __init__(self, text, token=None, tag=None):
        self.text = text
        self.token = token
        self.tag = tag[:2] if tag else None  # just first two letters of tag
        self.index = -1
        self.score = -1

    def __str__(self):
        return self.text + "\t" + self.token + "\n"

    def __repr__(self):
        return str(self)

class TextProcessor:
    """
    Pre-process text data to prepare for keyword extraction
    """

    def __init__(self):
        self.STOPWORDS = TextProcessor.__load_stopwords(path="../stopwords.txt")
        self.LEMMATIZER = WordNetLemmatizer()
        self.STEMMER = SnowballStemmer("english")
        self.PUNCTUATION = re.compile('([%s])+' % re.escape(string.punctuation), re.UNICODE)
        self.NUMERIC = re.compile(r"[0-9]+", re.UNICODE)
        self.PAT_ALPHABETIC = re.compile('(((?![\d])\w)+)', re.UNICODE)

    def remove_punctuation(self, s):
        """Removes punctuation from text"""
        return self.PUNCTUATION.sub(" ", s)

    def remove_numeric(self, s):
        """Removes numeric characters from text"""
        return self.NUMERIC.sub("", s)

    def remove_stopwords(self, tokens):
        """Removes stopwords from text"""
        return [w for w in tokens if w not in self.STOPWORDS]

    def stem_tokens(self, tokens):
        """Performs stemming on text data"""
        return [self.STEMMER.stem(word) for word in tokens]

    def lemmatize_tokens(self, tokens):
        """Performs lemmatization on text data using Part-of-Speech tags"""
        if not tokens:
            return []
        if isinstance(tokens[0], str):
            pos_tags = pos_tag(tokens)
        else:
            pos_tags = tokens
        tokens = [self.LEMMATIZER.lemmatize(word[0]) if not TextProcessor.__get_wordnet_pos(word[1])
                  else self.LEMMATIZER.lemmatize(word[0], pos=TextProcessor.__get_wordnet_pos(word[1]))
                  for word in pos_tags]
        return tokens

    def part_of_speech_tag(self, tokens):
        if isinstance(tokens, str):
            tokens = self.tokenize(tokens)
        return pos_tag(tokens)

    @staticmethod
    def __load_stopwords(path="stopwords.txt"):
        """Utility function to load stopwords from text file"""
        # with open(path, "r") as stopword_file:
        #     stopwords = [line.strip() for line in stopword_file.readlines()]
        return list(set(stopwords.words('english')))

    @staticmethod
    def __get_wordnet_pos(treebank_tag):
        """Maps the treebank tags to WordNet part of speech names"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    @staticmethod
    def deaccent(s):
        """Remove accentuation from the given string"""
        norm = unicodedata.normalize("NFD", s)
        result = "".join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
        return unicodedata.normalize("NFC", result)

    def clean_text(self, text, filters=None, stem=False):
        """ Tokenizes a given text into words, applying filters and lemmatizing them.
        Returns a dict of word -> SyntacticUnit"""
        text = text.lower()
        text = self.deaccent(text)
        text = self.remove_numeric(text)
        text = self.remove_punctuation(text)
        original_words = [match.group() for match in self.PAT_ALPHABETIC.finditer(text)]
        filtered_words = self.remove_stopwords(original_words)
        pos_tags = pos_tag(filtered_words)
        if stem:
            filtered_words = self.stem_tokens(filtered_words)
        else:
            filtered_words = self.lemmatize_tokens(pos_tags)
        units = []
        if not filters:
            filters = ['N', 'J']
        for i in range(len(filtered_words)):
            if not pos_tags[i][1].startswith('N') or len(filtered_words[i]) < 3:
                continue
            token = filtered_words[i]
            text = filtered_words[i]
            tag = pos_tags[i][1]
            sentence = SyntacticUnit(text, token, tag)
            sentence.index = i
            units.append(sentence)
        return {unit.text: unit for unit in units}

    def tokenize(self, text):
        """Performs basic preprocessing and tokenizes text data"""
        text = text.lower()
        text = self.deaccent(text)
        return [match.group() for match in self.PAT_ALPHABETIC.finditer(text)]

    def clean_sentence(self, text):
        """Cleans sentence for word2vec training"""
        text = text.lower()
        text = self.deaccent(text)
        text = self.remove_numeric(text)
        text = self.remove_punctuation(text)
        return text

class Graph:
    """
    Implementation of an undirected graph, based on Pygraph
    """

    WEIGHT_ATTRIBUTE_NAME = "weight"
    DEFAULT_WEIGHT = 0

    LABEL_ATTRIBUTE_NAME = "label"
    DEFAULT_LABEL = ""

    def __init__(self):
        # Metadata about edges
        self.edge_properties = {}    # Mapping: Edge -> Dict mapping, label-> str, wt->num
        self.edge_attr = {}          # Key value pairs: (Edge -> Attributes)
        # Metadata about nodes
        self.node_attr = {}          # Pairing: Node -> Attributes
        self.node_neighbors = {}     # Pairing: Node -> Neighbors

    def has_edge(self, edge):
        """Checks if a given edge exists in the graph"""
        u, v = edge
        return (u, v) in self.edge_properties and (v, u) in self.edge_properties

    def edge_weight(self, edge):
        """Returns the weight of the given edge"""
        return self.get_edge_properties(edge).setdefault(self.WEIGHT_ATTRIBUTE_NAME, self.DEFAULT_WEIGHT)

    def neighbors(self, node):
        """Returns a list of neighbors for a given node"""
        return self.node_neighbors[node]

    def has_node(self, node):
        """Checks if the grpah has a given node"""
        return node in self.node_neighbors

    def add_edge(self, edge, wt=1, label='', attrs=None):
        """Adds an edge to the graph"""
        if not attrs:
            attrs = []
        u, v = edge
        if v not in self.node_neighbors[u] and u not in self.node_neighbors[v]:
            self.node_neighbors[u].append(v)
            if u != v:
                self.node_neighbors[v].append(u)

            self.add_edge_attributes((u, v), attrs)
            self.set_edge_properties((u, v), label=label, weight=wt)
        else:
            raise ValueError("Edge (%s, %s) already in graph" % (u, v))

    def add_node(self, node, attrs=None):
        """Adds a node to the graph"""
        if attrs is None:
            attrs = []
        if node not in self.node_neighbors:
            self.node_neighbors[node] = []
            self.node_attr[node] = attrs
        else:
            raise ValueError("Node %s already in graph" % node)

    def nodes(self):
        """Returns a list of nodes in the graph"""
        return list(self.node_neighbors.keys())

    def edges(self):
        """Returns a list of edges in the graph"""
        return [a for a in list(self.edge_properties.keys())]

    def del_node(self, node):
        """Deletes a given node from the graph"""
        for each in list(self.neighbors(node)):
            if each != node:
                self.del_edge((each, node))
        del(self.node_neighbors[node])
        del(self.node_attr[node])

    # Helper methods
    def get_edge_properties(self, edge):
        """Returns the properties of an edge"""
        return self.edge_properties.setdefault(edge, {})

    def add_edge_attributes(self, edge, attrs):
        """Sets multiple edge attributes"""
        for attr in attrs:
            self.add_edge_attribute(edge, attr)

    def add_edge_attribute(self, edge, attr):
        """Sets a single edge attribute"""
        self.edge_attr[edge] = self.edge_attributes(edge) + [attr]

        if edge[0] != edge[1]:
            self.edge_attr[(edge[1], edge[0])] = self.edge_attributes((edge[1], edge[0])) + [attr]

    def edge_attributes(self, edge):
        """Returns edge attributes"""
        try:
            return self.edge_attr[edge]
        except KeyError:
            return []

    def set_edge_properties(self, edge, **properties):
        """Sets edge properties"""
        self.edge_properties.setdefault(edge, {}).update(properties)
        if edge[0] != edge[1]:
            self.edge_properties.setdefault((edge[1], edge[0]), {}).update(properties)

    def del_edge(self, edge):
        """Deletes an edge from the graph"""
        u, v = edge
        self.node_neighbors[u].remove(v)
        self.del_edge_labeling((u, v))
        if u != v:
            self.node_neighbors[v].remove(u)
            self.del_edge_labeling((v, u))

    def del_edge_labeling(self, edge):
        """Deletes the labeling of an edge from the graph"""
        keys = list(edge)
        keys.append(edge[::-1])

        for key in keys:
            for mapping in [self.edge_properties, self.edge_attr]:
                try:
                    del (mapping[key])
                except KeyError:
                    pass


class KeywordExtractor:
    """
    Extracts keywords from text using TextRank algorithm
    """

    def __init__(self, word2vec=None):
        self.preprocess = TextProcessor()
        self.graph = Graph()
        if word2vec:
            print("Loading word2vec embedding...")
            self.word2vec = KeyedVectors.load_word2vec_format(word2vec, binary=True)
            print("Succesfully loaded word2vec embeddings!")
        else:
            self.word2vec = None

    def init_graph(self):
        self.preprocess = TextProcessor()
        self.graph = Graph()

    def extract(self, text, ratio=0.4, split=False, scores=False):
        """
        :param: text: text data from which keywords are to be extracted
        :return: list of keywords extracted from text
        """
        self.init_graph()
        words = self.preprocess.tokenize(text)
        tokens = self.preprocess.clean_text(text)
        for word, item in tokens.items():
            if not self.graph.has_node(item.token):
                self.graph.add_node(item.token)
        self.__set_graph_edges(self.graph, tokens, words)
        del words
        KeywordExtractor.__remove_unreachable_nodes(self.graph)
        if len(self.graph.nodes()) == 0:
            return [] if split else ""
        pagerank_scores = self.__textrank()
        extracted_lemmas = KeywordExtractor.__extract_tokens(self.graph.nodes(), pagerank_scores, ratio)
        lemmas_to_word = KeywordExtractor.__lemmas_to_words(tokens)
        keywords = KeywordExtractor.__get_keywords_with_score(extracted_lemmas, lemmas_to_word)
        combined_keywords = self.__get_combined_keywords(keywords, text.split())
        return KeywordExtractor.__format_results(keywords, combined_keywords, split, scores)

    def __textrank(self, initial_value=None, damping=0.85, convergence_threshold=0.0001):
        """Implementation of TextRank on a undirected graph"""
        if not initial_value:
            initial_value = 1.0 / len(self.graph.nodes())
        scores = dict.fromkeys(self.graph.nodes(), initial_value)

        iteration_quantity = 0
        for iteration_number in range(100):
            iteration_quantity += 1
            convergence_achieved = 0
            for i in self.graph.nodes():
                rank = 1 - damping
                for j in self.graph.neighbors(i):
                    neighbors_sum = sum(self.graph.edge_weight((j, k)) for k in self.graph.neighbors(j))
                    rank += damping * scores[j] * self.graph.edge_weight((j, i)) / neighbors_sum
                if abs(scores[i] - rank) <= convergence_threshold:
                    convergence_achieved += 1
                scores[i] = rank
            if convergence_achieved == len(self.graph.nodes()):
                break
        return scores

    @staticmethod
    def __format_results(_keywords, combined_keywords, split, scores):
        """
        :param _keywords:dict of keywords:scores
        :param combined_keywords:list of word/s
        """
        combined_keywords.sort(key=lambda w: KeywordExtractor.__get_average_score(w, _keywords), reverse=True)
        if scores:
            return [(word, KeywordExtractor.__get_average_score(word, _keywords)) for word in combined_keywords]
        if split:
            return combined_keywords
        return "\n".join(combined_keywords)

    @staticmethod
    def __get_average_score(concept, _keywords):
        """Calculates average score"""
        word_list = concept.split()
        word_counter = 0
        total = 0
        for word in word_list:
            total += _keywords[word]
            word_counter += 1
        return total / word_counter

    def __strip_word(self, word):
        """Preprocesses given word"""
        stripped_word_list = list(self.preprocess.tokenize(word))
        return stripped_word_list[0] if stripped_word_list else ""

    def __get_combined_keywords(self, _keywords, split_text):
        """
        :param _keywords:dict of keywords:scores
        :param split_text: list of strings
        :return: combined_keywords:list
        """
        result = []
        _keywords = _keywords.copy()
        len_text = len(split_text)
        for i in range(len_text):
            word = self.__strip_word(split_text[i])
            if word in _keywords:
                combined_word = [word]
                if i + 1 == len_text:
                    result.append(word)  # appends last word if keyword and doesn't iterate
                for j in range(i + 1, len_text):
                    other_word = self.__strip_word(split_text[j])
                    if other_word in _keywords and other_word == split_text[j] \
                            and other_word not in combined_word:
                        combined_word.append(other_word)
                    else:
                        for keyword in combined_word:
                            _keywords.pop(keyword)
                        result.append(" ".join(combined_word))
                        break
        return result

    @staticmethod
    def __get_keywords_with_score(extracted_lemmas, lemma_to_word):
        """
        :param extracted_lemmas:list of tuples
        :param lemma_to_word: dict of {lemma:list of words}
        :return: dict of {keyword:score}
        """
        keywords = {}
        for score, lemma in extracted_lemmas:
            keyword_list = lemma_to_word[lemma]
            for keyword in keyword_list:
                keywords[keyword] = score
        return keywords

    @staticmethod
    def __lemmas_to_words(tokens):
        """Returns the corresponding words for the given lemmas"""
        lemma_to_word = {}
        for word, unit in tokens.items():
            lemma = unit.token
            if lemma in lemma_to_word:
                lemma_to_word[lemma].append(word)
            else:
                lemma_to_word[lemma] = [word]
        return lemma_to_word

    @staticmethod
    def __extract_tokens(lemmas, scores, ratio):
        lemmas.sort(key=lambda s: scores[s], reverse=True)
        length = len(lemmas) * ratio
        return [(scores[lemmas[i]], lemmas[i],) for i in range(int(length))]

    @staticmethod
    def __remove_unreachable_nodes(graph):
        for node in graph.nodes():
            if sum(graph.edge_weight((node, other)) for other in graph.neighbors(node)) == 0:
                graph.del_node(node)

    def __set_graph_edges(self, graph, tokens, words):
        self.__process_first_window(graph, tokens, words)
        self.__process_text(graph, tokens, words)

    def __process_first_window(self, graph, tokens, split_text):
        first_window = KeywordExtractor.__get_first_window(split_text)
        for word_a, word_b in combinations(first_window, 2):
            self.__set_graph_edge(graph, tokens, word_a, word_b)

    def __process_text(self, graph, tokens, split_text):
        queue = KeywordExtractor.__init_queue(split_text)
        for i in range(2, len(split_text)):
            word = split_text[i]
            self.__process_word(graph, tokens, queue, word)
            KeywordExtractor.__update_queue(queue, word)

    def __set_graph_edge(self, graph, tokens, word_a, word_b):
        if word_a in tokens and word_b in tokens:
            lemma_a = tokens[word_a].token
            lemma_b = tokens[word_b].token
            edge = (lemma_a, lemma_b)

            if graph.has_node(lemma_a) and graph.has_node(lemma_b) and not graph.has_edge(edge):
                if not self.word2vec:
                    graph.add_edge(edge)
                else:
                    try:
                        similarity = self.word2vec.similarity(lemma_a, lemma_b)
                        if similarity < 0:
                            similarity = 0.0
                    except:
                        similarity = 0.2
                    graph.add_edge(edge, wt=similarity)

    def __process_word(self, graph, tokens, queue, word):
        for word_to_compare in KeywordExtractor.__queue_iterator(queue):
            self.__set_graph_edge(graph, tokens, word, word_to_compare)

    @staticmethod
    def __get_first_window(split_text):
        return split_text[:2]

    @staticmethod
    def __init_queue(split_text):
        queue = Queue()
        first_window = KeywordExtractor.__get_first_window(split_text)
        for word in first_window[1:]:
            queue.put(word)
        return queue

    @staticmethod
    def __update_queue(queue, word):
        queue.get()
        queue.put(word)
        assert queue.qsize() == 1

    @staticmethod
    def __queue_iterator(queue):
        iterations = queue.qsize()
        for i in range(iterations):
            var = queue.get()
            yield var
            queue.put(var)
