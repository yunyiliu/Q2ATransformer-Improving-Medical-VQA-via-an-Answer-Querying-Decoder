import json

# 1. 拿到处理之后的close question class 和 open question
# 2. 把close question class 和 open quesions 过一个embedding layer
import numpy as np
from scipy import spatial
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
def glove():
    answers = json.load(open('../data/answer_query.json', 'r'))
    answers = answers["answers"]
    with open('../data/answer.txt', mode='w', encoding='utf-8') as f:
            f.write(answers)
    
    embeddings_dict = {}

    with open("/home/yunyi/yunyi/Modified_MedVQA-main/data/glove.6B.50d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    np.save('/home/yunyi/yunyi/Modified_MedVQA-main/data//wordsList', np.array(list(embeddings_dict.keys())))
    np.save('/home/yunyi/yunyi/Modified_MedVQA-main/data//wordVectors', np.array(list(embeddings_dict.values()), dtype='float32'))


    wordsList = np.load('/home/yunyi/yunyi/Modified_MedVQA-main/data/wordsList.npy')
    print('Loaded the word list!')

    wordsList = wordsList.tolist()  # Originally loaded as numpy array
    wordVectors = np.load('/home/yunyi/yunyi/Modified_MedVQA-main/data/wordVectors.npy')
    print('Loaded the word vectors!')
    print(len(wordsList))
    # print(wordsList)
    print(wordVectors.shape)

    baseballIndex = wordsList.index('baseball')
    print(baseballIndex)
    print(wordVectors[baseballIndex])



    
    
    embedding_res = []
    for answer in answers:
        answer_embedding = []
        answer = str(answer)
        
        answer_list = answer.split(" ")
        for word in answer_list:
            word = word.lower()
            if word=='semi-upright':
                w = word.split('-')
                word_index = wordsList.index(w[1])
            else:
                word_index = wordsList.index(word)
            word_embedding = wordVectors[word_index]
            answer_embedding.append(word_embedding)
        embedding_res.append(answer_embedding)


    answer_matrix = {'answer_embedding': embedding_res}
    print(embedding_res)
    json.dump(answer_matrix, open('../data/answer_matrix.json', 'w'))
    return embedding_res

if __name__ == '__main__':
    glove()
