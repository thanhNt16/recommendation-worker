import os
import sys
import random
import glob
import json
from bson.objectid import ObjectId
from threading import Thread

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

# Some utilites
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import pickle
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)
# Constant
DUMPED_MODEL = 'models/'

# connect mongo
client = MongoClient("localhost", 27017)
db = client.test
contents = db.contents

def train_content(user_id):
    print("### Start training")
    data = pd.DataFrame(list(contents.find({"customer": ObjectId(user_id)})))
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words=None)
    matrix = tf.fit_transform(data['content'])
    pickle.dump(matrix, open(DUMPED_MODEL + user_id + "_content.pickle", "wb")) 
    channel.basic_publish('', 'demo', 'training_complete')
    print(" [x] Sent to {0}: complete".format('demo'))


def callback(ch, method, properties, body):
    msg = body.decode()
    command = msg.split('_')[0]
    user_id = msg.split('_')[1]
    print(" [x] Received: ", body.decode())
    train_content(user_id)
    ch.basic_ack(method.delivery_tag)

# You can use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.mobilenet_v2 import MobileNetV2
# model = MobileNetV2(weights='imagenet')

print('Model loaded. Check http://127.0.0.1:5000/')

def course_recommend(name):
    idx = indices[name]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    course_indices = [i[0] for i in sim_scores]
    return course_title.iloc[course_indices]

def content_recommend(user_id):
    #Get course name
    course_name_list = exam_score_data[exam_score_data['userId'] == user_id].to_dict('records')
    if (len(course_name_list) != 0):
        course_name = course_name_list[0]['courseName']
        result = {
            'status': 'success',
            'data': course_recommend(course_name).head(10).to_dict('records')
        }
        return result
    else:
        return {
            'status': 'success',
            'data': 'User did not have any exam'
        }

@app.route('/', methods=['GET'])
def index():
    # Main page
    # customers = list(mongo.db.customers.find())
    # print(customers)
    data = pd.DataFrame(list(contents.find({ 'customer': ObjectId('603cfd4fe464cffdac1d18f9') })))
    print(data)
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words=None)
    matrix = tf.fit_transform(data['content'])
    # pickle.dump(matrix, open("vectorizer.pickle", "wb")) 
    # return render_template('index.html')
    # vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    # cosine_similarities = linear_kernel(vectorizer,matrix)
    # course_title = data[['_id','content']]
    # indices = pd.Series(data.index, index=data['content'])
    # idx = indices["Toy Story (1995)"]

    # sim_scores = list(enumerate(cosine_similarities[idx]))
    # sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # sim_scores = sim_scores[1:31]
    # course_indices = [i[0] for i in sim_scores]
    # print(course_title.iloc[course_indices])
    return 'hello'

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        is_new_user = False
        result = {
            'status': 'success',
            'data': 'User did not have any exam'
        }
        user_id = request.form.get('id')
        for user in user_data:
            if (user['id'] == user_id):
                is_new_user = isinstance(user['joinCourseIds'], list)
        is_new_user = False
        if (is_new_user):
            result = content_recommend(user_id)
        else:
            user_id_mapped = use_data[use_data['userId'] == user_id]['user_Id'].iloc[0]
            recommended = implicit_model.recommend(user_id_mapped, sparse_user_item,N = 20,filter_already_liked_items = False)
            courses = []
            scores = []
            courseIds = []

                # Get artist names from ids
            for item in recommended:
                idx, score = item
                courseIds.append(int(use_data.courseId.loc[use_data.item_Id == idx].iloc[0]))
                courses.append(use_data.courseName.loc[use_data.item_Id == idx].iloc[0])
                scores.append(score)

                # Create a dataframe of artist names and scores
            recommendations = pd.DataFrame({'id': courseIds, 'course': courses, 'score': scores}).to_dict('records')
            result = {
                'status': 'success',
                'data': recommendations
            }
        return jsonify({ "result": result })
    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)
     # connect rabbitmq
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.basic_consume(queue='test',
                      auto_ack=False,
                      on_message_callback=callback)
    print(' [*] Waiting for messages. To exit press CTRL+C')
    thread = Thread(target = channel.start_consuming)
    thread.start()

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
   
