from scipy import spatial


# function for recommending products for a certain product id
def get_recommendations(product_id, word2vecmodel):
    try:
        vector_array = word2vecmodel.wv.__getitem__([str(product_id)])
        vector_array = vector_array.flatten()
        product_id_list = list(word2vecmodel.wv.index_to_key)
        product_id_list.remove(product_id)

        similarity_list = []
        for prod_id in product_id_list:
            prod_array = word2vecmodel.wv.__getitem__([str(prod_id)])
            prod_array = prod_array.flatten()
            cosine_similarity = 1 - spatial.distance.cosine(vector_array, prod_array)
            similarity_list.append((prod_id, cosine_similarity))

        sorted_list = sorted(similarity_list, key=lambda x: (-x[1], x[0]))
        return sorted_list[:10]

    except KeyError as e:
        pass

