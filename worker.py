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
from train_data import train_data
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
                user_data = list(user_key[user_id])
                if (len(user_data) > 1):
                    test_list.append(user_data.pop(-1))
                    for x in user_data:
                        train_list.append(x)

            train = Interactions(train_list)
            test = Interactions(test_list)
            train.to_sequence(5, 3)
            inserted = train_collection.insert_many(train_list)
            # train_data.set_data(customer_id, train)

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
            print("### Training complete")
            # channel.basic_publish('', STATUS_QUEUE,
            #                       'complete|' + user_id + '|sequence')
            # print(" [x] Sent to {0}: complete_{1}".format(
            #     STATUS_QUEUE, user_id))
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
        publisher = Publisher(host='13.67.37.61',
                              username='rabbitmq',
                              password='rabbitmq',
                              queue=STATUS_QUEUE,
                              virtual_host='/')
        publisher.consume(queue=JOB_QUEUE, callback=callback)

    except:
        print("Unable to connect rabbitmq")
    
    http_server = WSGIServer(('0.0.0.0', 5001), app)
    http_server.serve_forever()
