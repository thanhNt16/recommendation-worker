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
# Mongo
import pymongo
from pymongo import MongoClient

# RabbitMQ
import pika
from publisher import Publisher

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
# import caser
from caser.train_caser import Recommender
from caser.interactions import Interactions
# from caser.train_caser import Recommender
# from caser.interactions import Interactions
import torch.optim as optim
import torch
from torch.autograd import Variable
import argparse

# Constant
DUMPED_MODEL = 'models/'
JOB_QUEUE = 'job_queue'
STATUS_QUEUE = 'status_queue'


# connect mongo
def getDb():
    try:
        mongo_uri = "mongodb+srv://harry:" + urllib.parse.quote(
            "Holmes@4869"
        ) + "@cluster0.ty1ik.mongodb.net/recsys?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE"
        client = MongoClient(mongo_uri)
        db = client.recsys
        return db
    except Exception as e:
        print("err", e)
        print("Unable to connect mongo")
        raise Exception("Unable to connect mongo")

def train_content(user_id):
    try:
        db = getDb()
        if (db):
            print("### Start training")
            contents = db.contents
            data = pd.DataFrame(list(contents.find({"customer": ObjectId(user_id)})))
            data.drop_duplicates()
            data.dropna()
            tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words=None)
            matrix = tf.fit_transform(data['content'])
            pickle.dump(matrix, open(DUMPED_MODEL + user_id + "_content.pickle", "wb")) 
            print("### Training complete")
            publisher.publish('complete|' + user_id + '|content')

            # channel.basic_publish('', STATUS_QUEUE, 'complete|' + user_id + '|content')
            print(" [x] Sent to {0}: complete_{1}".format(STATUS_QUEUE, user_id))
        else:
            raise Exception("Database not found")
    except Exception as e:
        print(e)
        raise Exception("Unable to connect mongo")

def train_caser(customer_id):
    try:
        db = getDb()
        if (db):
            print("### Start training")
            # inserted = train_collection.insert_many(train_list)
            parser = argparse.ArgumentParser()
            parser.add_argument(
                '--train_root',
                type=str,
                default='caser_pytorch/datasets/ml1m/test/train.txt')
            parser.add_argument(
                '--test_root',
                type=str,
                default='caser_pytorch/datasets/ml1m/test/test.txt')
            parser.add_argument('--L', type=int, default=5)
            parser.add_argument('--T', type=int, default=3)
            # train arguments
            parser.add_argument('--n_iter', type=int, default=5)
            parser.add_argument('--seed', type=int, default=1234)
            parser.add_argument('--batch_size', type=int, default=512)
            parser.add_argument('--learning_rate', type=float, default=1e-3)
            parser.add_argument('--l2', type=float, default=1e-6)
            parser.add_argument('--neg_samples', type=int, default=3)

            config = parser.parse_args()

            # model dependent arguments
            model_parser = argparse.ArgumentParser()
            model_parser.add_argument('--d', type=int, default=50)
            model_parser.add_argument('--nv', type=int, default=4)
            model_parser.add_argument('--nh', type=int, default=16)
            model_parser.add_argument('--drop', type=float, default=0.5)
            model_parser.add_argument('--ac_conv', type=str, default='relu')
            model_parser.add_argument('--ac_fc', type=str, default='relu')

            model_config = model_parser.parse_args()
            model_config.L = config.L

            sequences = db.sequences
            train_collection = db.train
            data = list(
                sequences.find({"customer": ObjectId(customer_id)}, {
                    '_id': False,
                    'customer': False,
                    'feedBack': False,
                    'createdAt': False,
                    'updatedAt': False,
                    'date': False
                }).sort('date', 1))
            user_ids = list(set(map(lambda row: row['userId'], data)))
            user_key = dict()

            for user_id in user_ids:
                user_key[user_id] = set()

            for row in data:
                user_key[row['userId']].add(json.dumps(row))

            train_list = list()
            test_list = list()
            

            for user_id in user_ids:
                if (len(user_key[user_id]) > 1):
                    user_data = user_key[user_id]
                    test_list.append(json.loads(user_data.pop()))
                    for x in user_data:
                        train_list.append(json.loads(x))
            # load dataset
            # print(len(train_list))

            # inserted = train_collection.insert_many(train_list)

            train = Interactions(train_list)
            # # transform triplets to sequence representation
            train.to_sequence(config.L, config.T)

            test = Interactions(test_list,
                                user_map=train.user_map,
                                item_map=train.item_map)
            model = Recommender(n_iter=50,
                                batch_size=512,
                                learning_rate=1e-3,
                                l2=1e-6,
                                neg_samples=3,
                                model_args=model_config,
                                use_cuda=False)

            model.fit(train, test, verbose=True)
            file_path = 'models/' + str(customer_id) + '_sequence'
            torch.save(model._net.state_dict(), file_path)
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
            data.drop_duplicates()
            data.dropna()
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
            publisher.publish('complete|' + user_id + '|collaborative_implicit')

            # channel.basic_publish('', STATUS_QUEUE, 'complete|' + user_id + '|collaborative_implicit')
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
            data.drop_duplicates()
            data.dropna()
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
            publisher.publish('complete|' + user_id + '|collaborative_explicit')

            # channel.basic_publish('', STATUS_QUEUE, 'complete|' + user_id + '|collaborative_explicit')
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

    if (algorithm == 'sequence'):
        ch.basic_ack(method.delivery_tag)
        train_caser(user_id)
        publisher.publish('complete|' + user_id + '|sequence')
        print(" [x] Sent to {0}: complete_{1}".format(STATUS_QUEUE, user_id))

app = Flask(__name__)


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)
    # connect rabbitmq

    try:
        publisher = Publisher(host='139.59.107.94',
                              username='rabbitmq',
                              password='rabbitmq',
                              queue=STATUS_QUEUE,
                              virtual_host='/')
        publisher.consume(queue=JOB_QUEUE, callback=callback)

    except:
        print("Unable to connect rabbitmq")
    
    http_server = WSGIServer(('0.0.0.0', 5001), app)
    http_server.serve_forever()
