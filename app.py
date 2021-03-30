import os
import sys
import random
import glob
import json
from bson.objectid import ObjectId
from threading import Thread
import dns
import urllib.parse 
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Mongo
import pymongo
from pymongo import MongoClient

# RabbitMQ
import pika

#sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 
import implicit

from surprise import Dataset
from surprise import Reader
from surprise import SVDpp, dump
from surprise.model_selection import GridSearchCV

# Some utilites
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import pickle
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)
# Constant
DUMPED_MODEL = 'models/'
JOB_QUEUE = 'job_queue'
STATUS_QUEUE = 'status_queue'

# connect mongo
def getDb():
    try:
        mongo_uri = "mongodb+srv://harry:" + urllib.parse.quote("Holmes@4869") + "@cluster0.ty1ik.mongodb.net/recsys?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE"
        client = MongoClient(mongo_uri)
        print(client.contents)
        db = client.recsys
        return db
    except:
        print("Unable to connect mongo")
        raise Exception("Unable to connect mongo")
   
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


def train_content(user_id):
    try:
        db = getDb()
        if (db):
            print("### Start training")
            contents = db.contents
            data = pd.DataFrame(list(contents.find({"customer": ObjectId(user_id)})))
            tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words=None)
            matrix = tf.fit_transform(data['content'])
            pickle.dump(matrix, open(DUMPED_MODEL + user_id + "_content.pickle", "wb")) 
            print("### Training complete")
            channel.basic_publish('', STATUS_QUEUE, 'complete|' + user_id + '|content')
            print(" [x] Sent to {0}: complete_{1}".format(STATUS_QUEUE, user_id))
        else:
            raise Exception("Database not found")
    except:
        raise Exception("Unable to connect mongo")


def train_collaborative_explicit(user_id):
    try:
        db = getDb()
        if (db):
            print("### Start training collaborative explicit")
            collaboratives = db.collaboratives
            data = pd.DataFrame(list(collaboratives.find({"customer": ObjectId(user_id), "explicit": True })))
            data = data[['userId','itemId', 'feedBack' ]]
            data = data.rename(columns={'userId': 'user', 'itemId': 'item', 'feedBack': 'rating'})
            lower_rating = data['rating'].min()
            upper_rating = data['rating'].max()

            reader = Reader(rating_scale=(lower_rating, upper_rating))

            data = Dataset.load_from_df(data[["user", "item", "rating"]], reader)
            svdpp = SVDpp(verbose = True, n_epochs = 5)
            svdpp.fit(data.build_full_trainset())
            file_name = os.path.expanduser('models/' + user_id + '_collaborative_explicit')
            dump.dump(file_name, algo=svdpp)

            # pickle.dump(matrix, open(DUMPED_MODEL + user_id + "_content.pickle", "wb")) 
            print("### Training collaborative explicit complete")
            channel.basic_publish('', STATUS_QUEUE, 'complete|' + user_id + '|collaborative_explicit')
            print(" [x] Sent to {0}: complete_{1}".format(STATUS_QUEUE, user_id))
        else:
            raise Exception("Database not found")
    except:
        raise Exception("Unable to connect mongo")


def train_collaborative_implicit(user_id):
    try:
        db = getDb()
        if (db):
            print("### Start training collaborative implicit")
            collaboratives = db.collaboratives
            data = pd.DataFrame(list(collaboratives.find({ 'customer': ObjectId(user_id), 'explicit': False  })))
            data = data[['userId','itemId', 'feedBack' ]]
            data = data.rename(columns={'userId': 'user', 'itemId': 'item'})
            data['user'] = data['user'].astype("category")
            data['item'] = data['item'].astype("category")

            #cat.codes creates a categorical id for the users and artists
            data['user_id'] = data['user'].cat.codes
            data['item_id'] = data['item'].cat.codes
            sparse_item_user = sparse.csr_matrix((data['feedBack'].astype(float), (data['item_id'], data['user_id'])))
            sparse_user_item = sparse.csr_matrix((data['feedBack'].astype(float), (data['user_id'], data['item_id'])))
            matrix_size = sparse_user_item.shape[0]*sparse_user_item.shape[1] # Number of possible interactions in the matrix
            num_purchases = len(sparse_user_item.nonzero()[0]) # Number of items interacted with
            sparsity = 100*(1 - (num_purchases/matrix_size))
            print(sparsity)
            model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=40)
            alpha_val = 15
            data_conf = (sparse_item_user * alpha_val).astype('double')

            model.fit(data_conf)
            file_name = os.path.expanduser('models/' + user_id + '_collaborative_implicit')
            dump.dump(file_name, algo=model)
            print("### Training complete")
            channel.basic_publish('', STATUS_QUEUE, 'complete|' + user_id + '|collaborative_implicit')
            print(" [x] Sent to {0}: complete_{1}".format(STATUS_QUEUE, user_id))
        else:
            raise Exception("Database not found")
    except:
        raise Exception("Unable to connect mongo")

def callback(ch, method, properties, body):
    msg = body.decode()
    msg = json.loads(msg)
    user_id = msg['user_id']
    command = msg['command']
    algorithm = msg['algorithm']
    params = msg['params']
    print(" [x] Received: ", body.decode())
    print(algorithm)
    if (algorithm == 'content'):
        train_content(user_id)
        ch.basic_ack(method.delivery_tag)
        
    if (algorithm == 'collaborative' and params == 'explicit'):
        # train_content(user_id)
        train_collaborative_explicit(user_id)
        ch.basic_ack(method.delivery_tag)

    if (algorithm == 'collaborative' and params == 'implicit'):
        # train_content(user_id)
        # train_collaborative_explicit(user_id)
        train_collaborative_implicit(user_id)
        ch.basic_ack(method.delivery_tag)
# You can use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.mobilenet_v2 import MobileNetV2
# model = MobileNetV2(weights='imagenet')

print('Model loaded. Check http://127.0.0.1:5000/')

@app.route('/', methods=['GET'])
def index():
    return 'Recommender API'

@app.route('/content', methods=['GET', 'POST'])
def recommendContent():
    if request.method == 'GET':
        try:
            db = getDb()
            if (db):
                contents = db.contents
                customer_id = request.args.get('customer_id', default = '')
                item_id = request.args.get('item_id', default = '')
                top = request.args.get('top', default = 10)
                product = contents.find_one({ 'itemId': item_id })
                
                data = pd.DataFrame(list(contents.find({ 'customer': ObjectId(customer_id) })))

                vectorizer = pickle.load(open("models/" + customer_id + "_content.pickle", "rb"))
                cosine_similarities = linear_kernel(vectorizer,vectorizer)
                course_title = data[['itemId','content']]
                indices = pd.Series(data.index, index=data['content'])
                idx = indices[product['content']]

                sim_scores = list(enumerate(cosine_similarities[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:int(top) + 1]
                course_indices = [i[0] for i in sim_scores]
                result = course_title.iloc[course_indices].to_dict('records')
                return { 'data': {
                    'current_product': {
                        'id': product['itemId'],
                        'content': product['content']
                    },
                    'similar_products': result,
                    'top': top
                } }
            
            else:
                return "Database not found"
        except Exception as e:
            return "Error in " + str(e)
    return None

@app.route('/collaborative_implicit', methods=['GET', 'POST'])
def recommend_collaborative_implicit():
    if request.method == 'GET':
        # try:
            db = getDb()
            if (db):
                collaboratives = db.collaboratives
                customer_id = request.args.get('customer_id', default = '')
                top = request.args.get('top', default = '')
                user_id = request.args.get('user_id', default = '')
                data = pd.DataFrame(list(collaboratives.find({ 'customer': ObjectId(customer_id), 'explicit': False  })))
                data = data[['userId','itemId', 'feedBack' ]]
                data = data.rename(columns={'userId': 'user', 'itemId': 'item'})
                data['user'] = data['user'].astype("category")
                data['item'] = data['item'].astype("category")

                # #cat.codes creates a categorical id for the users and artists
                data['user_id'] = data['user'].cat.codes
                data['item_id'] = data['item'].cat.codes
                sparse_item_user = sparse.csr_matrix((data['feedBack'].astype(float), (data['item_id'], data['user_id'])))
                sparse_user_item = sparse.csr_matrix((data['feedBack'].astype(float), (data['user_id'], data['item_id'])))
                user_ids =data[data['user'] == user_id].iloc[0]['user_id']

                _, model = dump.load('models/' + customer_id + '_collaborative_implicit')

                recommended = model.recommend(user_ids, sparse_user_item,N = int(top),filter_already_liked_items = False)
                result = []
                print('rec', recommended)
                for item in recommended:
                        idx, score = item
                        print('err', data[data.item_id == idx])
                        result.append({ 'item_id': str(data.item.loc[data.item_id == idx].iloc[0]), 'score': str(score) })
                return { 'data': {
                                'current_user': {
                                    'id': str(user_id),
                                },
                                'suggestion': result,
                                'top': top
                            } }
            else:
                return "Database not found"
        # except Exception as e:
        #     return "Error in " + str(e)

@app.route('/collaborative_explicit', methods=['GET', 'POST'])
def recommend_collaborative_explicit():
    if request.method == 'GET':
        try:
            db = getDb()
            if (db):
                collaboratives = db.collaboratives
                customer_id = request.args.get('customer_id', default = '')
                top = request.args.get('top', default = '')
                user_id = request.args.get('user_id', default = '')
                
                data = pd.DataFrame(list(collaboratives.find({ 'customer': ObjectId(customer_id), 'explicit': True })))
                data = data[['userId','itemId', 'feedBack' ]]
                data = data.rename(columns={'userId': 'user', 'itemId': 'item', 'feedBack': 'rating'})
                # # get list of product id
                iids = data['item'].unique()
                iids_user = data.loc[data['user'] == int(user_id), 'item']
                # # remove the idds that user has rated
                iids_to_pred = np.setdiff1d(iids, iids_user)
                testset = [[user_id, iid, 4.] for iid in iids_to_pred]
                _, loaded_algo = dump.load('models/' + customer_id + '_collaborative_explicit')
                
                predictions = loaded_algo.test(testset)
                pred_ratings = np.array([pred.est for pred in predictions])
                # i_max = pred_ratings.argmax()
                top_n = pred_ratings.argsort()[-int(top):][::-1]
                result = []

                for idx in top_n:
                    iid = iids_to_pred[idx]
                    result.append({ 'itemId': iid, 'prediction_rating': pred_ratings[idx] })
                return { 'data': {
                            'current_user': {
                                'id': user_id,
                            },
                            'suggestion': result,
                            'top': top
                        } }
            else:
                return "Database not found"
        except Exception as e:
            return "Error in " + str(e)
if __name__ == '__main__':
    # app.run(port=5002, threaded=False)
     # connect rabbitmq
    try:
        credentials = pika.PlainCredentials('rabbitmq', 'rabbitmq')
        connection = pika.BlockingConnection(pika.ConnectionParameters('13.67.37.61', 5672, '/', credentials))
        channel = connection.channel()
    # channel.queue_declare(JOB_QUEUE, durable=True)
    # channel.queue_declare(STATUS_QUEUE, durable=True)
    except:
        print("Unable to connect rabbitmq")

    channel.basic_consume(queue=JOB_QUEUE,
                      auto_ack=False,
                      on_message_callback=callback)
    print(' [*] Waiting for messages. To exit press CTRL+C')
    thread = Thread(target = channel.start_consuming)
    thread.start()

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
   
