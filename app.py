import os
import sys
import random
import glob
import json
from bson.objectid import ObjectId
from tqdm import tqdm
from threading import Thread
import urllib.parse
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_cors import CORS

# Mongo
import pymongo
from pymongo import MongoClient
from train_data import train_data
# RabbitMQ
import pika
from publisher import Publisher
import matplotlib.pyplot as plt

#sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans

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
# import caser
from caser.train_caser import Recommender
from caser.interactions import Interactions
import torch.optim as optim
import torch
from torch.autograd import Variable
import argparse
from popularity_recommender.model import popularity_recommender_py

train_dict = dict()

# Declare a flask app
app = Flask(__name__)
CORS(app)

# Constant
# DUMPED_MODEL = 'models/'
# JOB_QUEUE = 'job_queue'
# STATUS_QUEUE = 'status_queue'
# train_dict = dict()


# connect mongo
def getDb():
    try:
        mongo_uri = "mongodb+srv://harry:" + urllib.parse.quote(
            "Holmes@4869"
        ) + "@cluster0.ty1ik.mongodb.net/recsys?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE"
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


def train_caser(customer_id):
    try:
        db = getDb()
        if (db):
            print("### Start training")
            model_config = argparse.Namespace()
            model_config.d = 50
            model_config.nv = 4
            model_config.nh = 16
            model_config.drop = 0.5
            model_config.ac_conv = 'relu'
            model_config.ac_fc = 'relu'
            model_config.L = 5
            print("config", model_config)

            sequences = db.sequences
            data = list(sequences.find({"customer": ObjectId(customer_id)}))
            user_ids = list(set(map(lambda row: row['userId'], data)))

            user_key = dict()
            for user_id in tqdm(user_ids):
                user_key[user_id] = list()
                for row in data:
                    if (row['userId'] == user_id):
                        user_key[user_id].append(row)

            train_list = list()
            test_list = list()
            for user_id in tqdm(user_ids):
                publisher._conn.process_data_events()
                user_data = list(user_key[user_id])
                if (len(user_data) > 1):
                    test_list.append(user_data.pop(-1))
                    for x in user_data:
                        train_list.append(x)

            train = Interactions(train_list)
            test = Interactions(test_list)
            train.to_sequence(5, 3)

            train_dict[customer_id] = train

            model = Recommender(n_iter=5,
                                batch_size=512,
                                learning_rate=1e-3,
                                l2=1e-6,
                                neg_samples=3,
                                model_args=model_config,
                                use_cuda=False)

            model.fit(train, test, verbose=True)
            file_path = 'models/' + str(customer_id) + '_sequence'
            print("us", customer_id, file_path)
            torch.save(model._net.state_dict(), file_path)
            print("### Training complete")
            # channel.basic_publish('', STATUS_QUEUE,
            #                       'complete|' + user_id + '|sequence')
            # print(" [x] Sent to {0}: complete_{1}".format(
            #     STATUS_QUEUE, user_id))
        else:
            raise Exception("Database not found")
    except:
        raise Exception("Unable to connect mongo")


def train_content(user_id):
    try:
        db = getDb()
        if (db):
            print("### Start training")
            contents = db.contents
            data = pd.DataFrame(
                list(contents.find({"customer": ObjectId(user_id)})))
            tf = TfidfVectorizer(analyzer='word',
                                 ngram_range=(1, 3),
                                 min_df=0,
                                 stop_words=None)
            matrix = tf.fit_transform(data['content'])
            pickle.dump(matrix,
                        open(DUMPED_MODEL + user_id + "_content.pickle", "wb"))
            print("### Training complete")
            publisher.publish('complete|' + user_id + '|content')

            # channel.basic_publish('', STATUS_QUEUE, 'complete|' + user_id + '|content')
            print(" [x] Sent to {0}: complete_{1}".format(
                STATUS_QUEUE, user_id))
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
            data = pd.DataFrame(
                list(
                    collaboratives.find({
                        "customer": ObjectId(user_id),
                        "explicit": True
                    })))
            data = data[['userId', 'itemId', 'feedBack']]
            data = data.rename(columns={
                'userId': 'user',
                'itemId': 'item',
                'feedBack': 'rating'
            })
            lower_rating = data['rating'].min()
            upper_rating = data['rating'].max()

            reader = Reader(rating_scale=(lower_rating, upper_rating))

            data = Dataset.load_from_df(data[["user", "item", "rating"]],
                                        reader)
            svdpp = SVDpp(verbose=True, n_epochs=5)
            svdpp.fit(data.build_full_trainset())
            file_name = os.path.expanduser('models/' + user_id +
                                           '_collaborative_explicit')
            dump.dump(file_name, algo=svdpp)

            # pickle.dump(matrix, open(DUMPED_MODEL + user_id + "_content.pickle", "wb"))
            print("### Training collaborative explicit complete")
            publisher.publish('complete|' + user_id +
                              '|collaborative_explicit')

            # channel.basic_publish('', STATUS_QUEUE, 'complete|' + user_id + '|collaborative_explicit')
            print(" [x] Sent to {0}: complete_{1}".format(
                STATUS_QUEUE, user_id))
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
            data = pd.DataFrame(
                list(
                    collaboratives.find({
                        'customer': ObjectId(user_id),
                        'explicit': False
                    })))
            data = data[['userId', 'itemId', 'feedBack']]
            data = data.rename(columns={'userId': 'user', 'itemId': 'item'})
            data['user'] = data['user'].astype("category")
            data['item'] = data['item'].astype("category")

            #cat.codes creates a categorical id for the users and artists
            data['user_id'] = data['user'].cat.codes
            data['item_id'] = data['item'].cat.codes
            sparse_item_user = sparse.csr_matrix(
                (data['feedBack'].astype(float), (data['item_id'],
                                                  data['user_id'])))
            sparse_user_item = sparse.csr_matrix(
                (data['feedBack'].astype(float), (data['user_id'],
                                                  data['item_id'])))
            matrix_size = sparse_user_item.shape[0] * sparse_user_item.shape[
                1]  # Number of possible interactions in the matrix
            num_purchases = len(sparse_user_item.nonzero()
                                [0])  # Number of items interacted with
            sparsity = 100 * (1 - (num_purchases / matrix_size))
            print(sparsity)
            model = implicit.als.AlternatingLeastSquares(factors=20,
                                                         regularization=0.1,
                                                         iterations=40)
            alpha_val = 15
            data_conf = (sparse_item_user * alpha_val).astype('double')

            model.fit(data_conf)
            file_name = os.path.expanduser('models/' + user_id +
                                           '_collaborative_implicit')
            dump.dump(file_name, algo=model)
            print("### Training complete")
            publisher.publish('complete|' + user_id +
                              '|collaborative_implicit')

            # channel.basic_publish('', STATUS_QUEUE, 'complete|' + user_id + '|collaborative_implicit')
            print(" [x] Sent to {0}: complete_{1}".format(
                STATUS_QUEUE, user_id))
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

    if (algorithm == 'sequence'):
        ch.basic_ack(method.delivery_tag)
        train_caser(user_id)
        publisher.publish('complete|' + user_id + '|sequence')
        print(" [x] Sent to {0}: complete_{1}".format(STATUS_QUEUE, user_id))

        # channel.basic_publish('', STATUS_QUEUE, 'complete|' + user_id + '|sequence')


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.mobilenet_v2 import MobileNetV2
# model = MobileNetV2(weights='imagenet')

print('Model loaded. Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET'])
def index():
    try:
        db = getDb()
        if (db):
            customer_id = '60a34ac56e2c8b00201b3217' 
            print("### Start training")
            model_config = argparse.Namespace()
            model_config.d = 50
            model_config.nv = 4
            model_config.nh = 16
            model_config.drop = 0.5
            model_config.ac_conv = 'relu'
            model_config.ac_fc = 'relu'
            model_config.L = 5
            print("config", model_config)

            sequences = db.sequences
            train_collection = db.train
            data = list(sequences.find({"customer": ObjectId(customer_id)}))
            # sequences.delete_many({"customer": ObjectId(customer_id)})
            user_ids = list(set(map(lambda row: row['userId'], data)))
            print('data',  data, customer_id)
            user_key = dict()
            for user_id in tqdm(user_ids):
                user_key[user_id] = list()
                for row in data:
                    if (row['userId'] == user_id):
                        user_key[user_id].append(row)

            train_list = list()
            test_list = list()
            for user_id in tqdm(user_ids):
                user_data = list(user_key[user_id])
                if (len(user_data) > 1):
                    test_list.append(user_data.pop(-1))
                    for x in user_data:
                        train_list.append(x)

            train = Interactions(train_list)
            test = Interactions(test_list)
            train.to_sequence(5, 3)
            # inserted = train_collection.insert_many(train_list)
            # train_data.set_data(customer_id, train)

            model = Recommender(n_iter=5,
                                batch_size=512,
                                learning_rate=1e-3,
                                l2=1e-6,
                                neg_samples=3,
                                model_args=model_config,
                                use_cuda=False)

            model.fit(train, test, verbose=True)
            file_path = 'models/' + str(customer_id) + '_sequence'
            torch.save(model._net.state_dict(), file_path)
            print("### Training complete")

    except Exception as e:
        return "Error in " + str(e)


@app.route('/popular', methods=['GET'])
def popular():
    try:
        db = getDb()
        if (db):
            top = request.args.get('top', default=10)
            sequence_collection = db.sequences
            train_data = data = pd.DataFrame(list(sequence_collection.find(
                {})))
            popular_model = popularity_recommender_py()
            popular_model.create(train_data, 'itemId', 'feedBack')
            list_items = popular_model.recommend(top=int(top))
            # response
            response = jsonify(data={
                'popular_items': list_items,
                'top': str(top)
            })
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response
            # return {'data': {'popular_items': list_items, 'top': str(top)}}
    except Exception as e:
        return "Error in " + str(e)


@app.route('/content', methods=['GET', 'POST'])
def recommendContent():
    if request.method == 'GET':
        try:
            db = getDb()
            if (db):
                contents = db.contents
                customer_id = request.args.get('customer_id', default=None)
                item_id = request.args.get('item_id', default=None)
                top = request.args.get('top', default=10)
                product = contents.find_one({'itemId': item_id})
                if (item_id == None):
                    return "Error. Not found item_id"
                if (customer_id == None):
                    return "Error. Not found customer_id"
                data = pd.DataFrame(
                    list(contents.find({'customer': ObjectId(customer_id)})))

                vectorizer = pickle.load(
                    open("models/" + customer_id + "_content.pickle", "rb"))
                cosine_similarities = linear_kernel(vectorizer, vectorizer)
                course_title = data[['itemId', 'content']]
                indices = pd.Series(data.index, index=data['content'])
                idx = indices[product['content']]

                sim_scores = list(enumerate(cosine_similarities[idx]))

                sim_scores = sorted(sim_scores,
                                    key=lambda x: x[1],
                                    reverse=True)

                sim_scores = sim_scores[1:int(top) + 1]
                course_indices = [i[0] for i in sim_scores]
                result = course_title.iloc[course_indices].to_dict('records')
                # result = []
                print(result, course_indices)
                response = jsonify(
                    data={
                        'current_product': {
                            'id': product['itemId'],
                            'content': product['content']
                        },
                        'similar_products': result,
                        'top': top
                    })
                response.headers.add("Access-Control-Allow-Origin", "*")
                return response
                # return {
                #     'data': {
                #         'current_product': {
                #             'id': product['itemId'],
                #             'content': product['content']
                #         },
                #         'similar_products': result,
                #         'top': top
                #     }
                # }

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
            customer_id = request.args.get('customer_id', default='')
            top = request.args.get('top', default='')
            user_id = request.args.get('user_id', default='')
            data = pd.DataFrame(
                list(
                    collaboratives.find({
                        'customer': ObjectId(customer_id),
                        'explicit': False
                    })))
            data = data[['userId', 'itemId', 'feedBack']]
            data = data.rename(columns={'userId': 'user', 'itemId': 'item'})
            data['user'] = data['user'].astype("category")
            data['item'] = data['item'].astype("category")

            # #cat.codes creates a categorical id for the users and artists
            data['user_id'] = data['user'].cat.codes
            data['item_id'] = data['item'].cat.codes
            sparse_item_user = sparse.csr_matrix(
                (data['feedBack'].astype(float), (data['item_id'],
                                                  data['user_id'])))
            sparse_user_item = sparse.csr_matrix(
                (data['feedBack'].astype(float), (data['user_id'],
                                                  data['item_id'])))
            user_ids = data[data['user'] == user_id].iloc[0]['user_id']

            _, model = dump.load('models/' + customer_id +
                                 '_collaborative_implicit')

            recommended = model.recommend(user_ids,
                                          sparse_user_item,
                                          N=int(top),
                                          filter_already_liked_items=False)
            result = []
            for item in recommended:
                idx, score = item
                print('err', data[data.item_id == idx])
                result.append({
                    'item_id':
                    str(data.item.loc[data.item_id == idx].iloc[0]),
                    'score':
                    str(score)
                })

            response = jsonify(
                data={
                    'current_user': {
                        'id': str(user_id),
                    },
                    'suggestion': result,
                    'top': top
                })
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response
            # return {
            #     'data': {
            #         'current_user': {
            #             'id': str(user_id),
            #         },
            #         'suggestion': result,
            #         'top': top
            #     }
            # }
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
                customer_id = request.args.get('customer_id', default='')
                top = request.args.get('top', default='')
                user_id = request.args.get('user_id', default='')

                data = pd.DataFrame(
                    list(
                        collaboratives.find({
                            'customer': ObjectId(customer_id),
                            'explicit': True
                        })))
                data = data[['userId', 'itemId', 'feedBack']]
                data = data.rename(columns={
                    'userId': 'user',
                    'itemId': 'item',
                    'feedBack': 'rating'
                })
                # # get list of product id
                iids = data['item'].unique()
                iids_user = data.loc[data['user'] == int(user_id), 'item']
                # # remove the idds that user has rated
                iids_to_pred = np.setdiff1d(iids, iids_user)
                testset = [[user_id, iid, 4.] for iid in iids_to_pred]
                _, loaded_algo = dump.load('models/' + customer_id +
                                           '_collaborative_explicit')

                predictions = loaded_algo.test(testset)
                pred_ratings = np.array([pred.est for pred in predictions])
                # i_max = pred_ratings.argmax()
                top_n = pred_ratings.argsort()[-int(top):][::-1]
                result = []

                for idx in top_n:
                    iid = iids_to_pred[idx]
                    result.append({
                        'itemId': iid,
                        'prediction_rating': pred_ratings[idx]
                    })

                response = jsonify(
                    data={
                        'current_user': {
                            'id': user_id,
                        },
                        'suggestion': result,
                        'top': top
                    })
                response.headers.add("Access-Control-Allow-Origin", "*")
                return response
                # return {
                #     'data': {
                #         'current_user': {
                #             'id': user_id,
                #         },
                #         'suggestion': result,
                #         'top': top
                #     }
                # }
            else:
                return "Database not found"
        except Exception as e:
            return "Error in " + str(e)


@app.route('/sequence', methods=['GET', 'POST'])
def recommend_sequence():
    if request.method == 'GET':
        db = getDb()
        if (db):
            customer_id = request.args.get('customer_id', default='')
            top = request.args.get('top', default='')
            user_id = request.args.get('user_id', default='')
            train_collection = db.train
            train_list = list(
                train_collection.find({'customer': ObjectId(customer_id)}))
            train = Interactions(train_list)
            train.to_sequence(5, 3)

            model_config = argparse.Namespace()
            model_config.d = 50
            model_config.nv = 4
            model_config.nh = 16
            model_config.drop = 0.5
            model_config.ac_conv = 'relu'
            model_config.ac_fc = 'relu'
            model_config.L = 5
            print("config", model_config)
            model = Recommender(n_iter=5,
                                batch_size=512,
                                learning_rate=1e-3,
                                l2=1e-6,
                                neg_samples=3,
                                model_args=model_config,
                                use_cuda=False)
            model._initialize(train)
            model._net.load_state_dict(
                torch.load('models/' + customer_id + '_sequence'))
            userid = train.user_map[user_id]
            print("user", user_id)
            scores = model.predict(userid)
            item_ids = np.unique(train.item_ids)
            result = list()

            for idx, item_id in enumerate(item_ids):
                item = list(train.item_map.keys())[list(
                    train.item_map.values()).index(item_id)]

                result.append({'id': int(item), 'score': int(scores[idx])})

            result = sorted(result, key=lambda k: k['score'])
            
            response = jsonify(
                data={
                    'current_user': {
                        'id': user_id,
                    },
                    'suggestion': result[-int(top):],
                    'top': int(top)
                })
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response
            # return {
            #     'data': {
            #         'current_user': {
            #             'id': user_id,
            #         },
            #         'suggestion': result[-int(top):],
            #         'top': int(top)
            #     }
            # }


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)
    # connect rabbitmq
    # try:
    #     publisher = Publisher(host='13.67.37.61',username='rabbitmq', password='rabbitmq', queue=STATUS_QUEUE, virtual_host='/')
    #     publisher.consume(queue=JOB_QUEUE, callback=callback)

    #     # credentials = pika.PlainCredentials('rabbitmq', 'rabbitmq')
    #     # connection = pika.BlockingConnection(pika.ConnectionParameters('13.67.37.61', 5672, '/', credentials))
    #     # channel = connection.channel()
    # # channel.queue_declare(JOB_QUEUE, durable=True)
    # # channel.queue_declare(STATUS_QUEUE, durable=True)
    # except:
    #     print("Unable to connect rabbitmq")

    # channel.basic_consume(queue=JOB_QUEUE,
    #                   auto_ack=False,
    #                   on_message_callback=callback)
    # print(' [*] Waiting for messages. To exit press CTRL+C')
    # thread = Thread(target = channel.start_consuming)
    # thread.start()

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
