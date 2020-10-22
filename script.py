import subprocess
import sys
from sentence_transformers import util
import numpy as np

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('sentence_transformers')

clean_corpus = list()

clean_corpus = list()
def add_to_corpus(file): 
  with open(file, "r") as f:
    corpus = f.readlines()
  for sentence in corpus: 
    if sentence.lower() in clean_corpus: 
      continue
    elif sentence.lower() not in clean_corpus:
      clean_corpus.append(sentence.lower())
  return clean_corpus

add_to_corpus("common_phrases.txt")
print(len(clean_corpus))
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

import torch 
top_k = 10
corpus_torch = torch.load('corpus.pt')

while True: 
    text = input("Type your query here (type exit to escape): ")
    if text.lower() == 'exit': 
        break
    else: 
        queries = [text]
        for query in queries:
            query_embedding = embedder.encode(query, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_embedding, corpus_torch)[0]
            cos_scores = cos_scores.cpu()

            #We use np.argpartition, to only partially sort the top_k results
            top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

            print("\n\n======================\n\n")
            print("Query:", query)
            print("\nTop 10 most similar sentences in corpus:")

            for idx in top_results[0:top_k]:
              if round(cos_scores[idx].item(), 4) == 1.0000:
                continue
              else: 
                print(clean_corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))