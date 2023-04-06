# %%

# %%
# Import Libarary Dependency
import os
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
# %%
# Data Construct
database = {
    "clause":[
        'The student who studies hard usually gets good grades.',
        'The teacher who explains clearly makes the lesson easier to understand.',
        'The cat that sleeps all day has become very lazy.',
        'The store where they sell fresh vegetables is just around the corner.',
        'The book that I borrowed from the library was very interesting.',
        'The movie that we watched last night was quite thrilling.',
        'The restaurant which serves delicious food is always crowded.',
        'The museum that exhibits historical artifacts attracts many visitors.',
        'The athlete who trains consistently often achieves success.',
        'The friend who listens attentively is always appreciated.',
    ],
    'present_perfect':[
        'I have lived in this city for five years.',
        'She has traveled to several countries.',
        'They have finished their homework.',
        'We have visited the museum many times.',
        'He has never eaten sushi before.',
        'The cat has caught three mice today.',
        'The students have studied this topic already.',
        'She has worked here since 2018.',
        'The team has won three championships.',
        'I have read this book twice.',
    ],
    'comparative': [
        'This car is faster than the other one.',
        'She is taller than her sister.',
        'My house is smaller than yours.',
        'This pizza is more delicious than the one we had last week.',
        'Studying English is easier than learning Chinese for some people.',
        'This computer is less expensive than that one.',
        'The weather today is hotter than yesterday.',
        'This movie is more interesting than the one we saw last time.',
        'His new bike is lighter than his old one.',
        'The test was more difficult than I expected.',
    ],
    'simple_future': [
        'I will go to the supermarket later.',
        'She will travel to Europe next year.',
        'They will attend the party this weekend.',
        'We will finish the project by the end of the month.',
        'He will call you when he arrives.',
        'The teacher will correct the tests tomorrow.',
        'The concert will start at 8 pm.',
        'She will cook dinner tonight.',
        'They will move to a new house next month.',
        'I will send you an email with the details.',
    ]
}
# %%
def extract_subtree(token):
    if not list(token.children):
        return f"{token.pos_}"
    else:
        children_str = " ".join([extract_subtree(child) for child in token.children])
        return f"{token.pos_}[{children_str}]"
def sentences_convert2dependency(sentences):
    # en_nlp = spacy.load('en_core_web_sm')
    # result = []
    # for sentence in sentences:
    #     doc = en_nlp(sentence)
    #     dep = []
    #     for token in doc:
    #         dep.append(token.dep_)
    #     result.append(" ".join(dep))
    # return result

    en_nlp = spacy.load('en_core_web_sm')
    result = []
    for sentence in sentences:
        doc = en_nlp(sentence)
        root_token = [token for token in doc if token.head == token][0]
        tree_str = extract_subtree(root_token)
        result.append(tree_str)
    return result

def sentence_convert2dependency(sentence):
    # en_nlp = spacy.load('en_core_web_sm')
    # doc = en_nlp(sentence)
    # dep = []
    # for token in doc:
    #     dep.append(token.dep_)
    # result = (" ".join(dep))
    # return result

    en_nlp = spacy.load('en_core_web_sm')
    doc = en_nlp(sentence)
    root_token = [token for token in doc if token.head == token][0]
    tree_str = extract_subtree(root_token)
    result = tree_str
    return result
# %%
# Convert sentence to its dependency
en_nlp = spacy.load('en_core_web_sm')
dependency_database = {}
for key in database:
    dependency_database[key] = sentences_convert2dependency(database[key])

database_sentences = []
for key in database:
    database_sentences += database[key]
    

# %%
'''
Define Faiss Object
'''
class FaissEngine:
    def __init__(self, database, model, layer_num=3, max_length=20):
        self.database = database
        self.database_sentences = []
        for key in self.database:
            self.database_sentences += self.database[key]
        self.model = model
        self.layer_num = layer_num
        self.max_length = 20
        self.database_embeddings = self.encode_sentence(self.database_sentences)
        self.index = faiss.IndexFlatL2(self.database_embeddings.shape[1])
        self.index.add(np.array(self.database_embeddings))

    def encode_sentence(self, sentence):
        model_input = self.model.tokenizer(sentence, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
        input_ids = model_input['input_ids']
        attention_mask = model_input['attention_mask']
        
        with torch.no_grad():
            # Get all hidden states from the model
            outputs = self.model._first_module().auto_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # Extract the specific layer's output
            hidden_states = outputs.hidden_states[self.layer_num + 1]        # Use attention_mask to ignore [PAD] tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        masked_hidden_states = hidden_states * mask_expanded
        sentence_embedding = torch.sum(masked_hidden_states, dim=1) / torch.clamp(torch.sum(mask_expanded, dim=1), min=1e-9)
        return sentence_embedding

    def search_similar_sentences(self, input_sentence, k=3):
        input_sentence = sentence_convert2dependency(input_sentence)
        input_embedding = self.encode_sentence([input_sentence])

        distances, indices = self.index.search(np.array(input_embedding), k)
        # return [(self.database_sentences[i], d) for i, d in zip(indices[0], distances[0])]
        
        return [(database_sentences[i], self.database_sentences[i], d) for i, d in zip(indices[0], distances[0])]
    
    def search(self, input_sentences, k=10):
        similar_sentences_sets = []
        for sentence in input_sentences:
            sentence = sentence_convert2dependency(sentence)
            query_embedding = self.encode_sentence([sentence])
            _, indices = self.index.search(np.array(query_embedding), k)
            similar_sentences_sets.append(set(indices.flatten()))

        # Find sentences that are similar to both input sentences
        # 取出取 input_sentences 都有交集的 similar sentences
        common_sentences = set.intersection(*similar_sentences_sets)
        print("Sentences that are similar to both input sentences:")
        for idx in common_sentences:
            print(database_sentences[idx])
# %%
model_name = 'all-mpnet-base-v2'
model = SentenceTransformer(model_name)
# %%
faiss_engine = FaissEngine(dependency_database, model, layer_num=2)
# %%
# Example sentence
sentence = "The student who studies hard usually gets good grades."
# Encode the sentence and get the specific layer output as embedding
# faiss_engine.search_similar_sentences(sentence, k=5)
faiss_engine.search(['I will go to the supermarket later.', \
    'The teacher will correct the tests tomorrow.', 'I will send you an email with the details.'])
# %%
