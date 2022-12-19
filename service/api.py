from flask import Flask
from flask_restful import Resource, Api
from functions.recommender import get_recommendations
import pickle


app = Flask(__name__)
api = Api(app)

model = pickle.load(open('service/model.pickle', 'rb'))


class Recommendations(Resource):

    def get(self, product_id):
        product_id = str(product_id)
        recommendations = get_recommendations(product_id=product_id, word2vecmodel=model)
        if recommendations is None:
            return {product_id: None}, 404
        else:
            return {product_id: recommendations}


api.add_resource(Recommendations, '/recommendations/<string:product_id>')


if __name__ == '__main__':
    app.run()

    