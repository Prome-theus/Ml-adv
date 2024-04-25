from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd 
import sklearn as sk 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

