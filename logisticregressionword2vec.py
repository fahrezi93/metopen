import pickle
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from gensim.models import Word2Vec

# Load dataset
file_dataset = './metopen/dataset_tweet_sentiment_pilkada_DKI_2017.csv'
data = pd.read_csv(file_dataset)

# Preprocessing functions
def remove_at_hash(sent):
    return re.sub(r'@|#', r'', sent.lower())

def remove_sites(sent):
    return re.sub(r'http.*', r'', sent.lower())

def remove_punct(sent):
    return ' '.join(re.findall(r'\w+', sent.lower()))

# Apply preprocessing
data['text'] = data['Text Tweet'].apply(lambda x: remove_punct(remove_sites(remove_at_hash(x))))

# Encode labels
le = preprocessing.LabelEncoder()
le.fit(data['Sentiment'])
data['label'] = le.transform(data['Sentiment'])

# Split data
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train)
tfidf_test_vectors = tfidf_vectorizer.transform(X_test)

# Feature Selection
fs = True
if fs:
    fs_label = "ChiSquare"
    ch2 = SelectKBest(chi2, k=900)
    tfidf_train_vectors = ch2.fit_transform(tfidf_train_vectors, y_train)
    tfidf_test_vectors = ch2.transform(tfidf_test_vectors)
else:
    fs_label = "None"

# SVM with gamma
pKernel = ['linear', 'rbf']
pC = [0.1, 1.0, 10.0]
pGamma = [0.01, 0.1, 1.0]
ik = 0
ic = 0
ig = 0

svm_classifier = svm.SVC(kernel=pKernel[ik], C=pC[ic], gamma=pGamma[ig])
svm_classifier.fit(tfidf_train_vectors, y_train)
y_pred = svm_classifier.predict(tfidf_test_vectors)
print(classification_report(y_test, y_pred))
cnf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix (TN, FP, FN, TP):')
print(cnf_matrix)

group_names = ['TN', 'FP', 'FN', 'TP']
group_counts = ["{0:0.0f}".format(value) for value in cnf_matrix.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cnf_matrix, annot=labels, fmt='', cmap='Blues')
#plt.show()

print(f'Sel. Fitur\t: {fs_label}')
print(f'Param. SVM\t: Kernel={pKernel[ik]}, C={pC[ic]}, Gamma={pGamma[ig]}')
print(f'Jml. Data\t: {tfidf_train_vectors.shape[0]} (80%)')
print(f'Jml. Fitur\t: {tfidf_train_vectors.shape[1]}')
print('Precision\t: {:.2}'.format(precision_score(y_test, y_pred)))
print('Recall\t\t: {:.2}'.format(recall_score(y_test, y_pred)))
print('Accuracy\t: {:.2}'.format(accuracy_score(y_test, y_pred)))
print('F1-Score\t: {:.2}'.format(f1_score(y_test, y_pred)))

filename = f'model-svm-{fs_label}-{pKernel[ik]}-{pC[ic]}-{pGamma[ig]}.pickle'
pickle.dump(svm_classifier, open(filename, 'wb'))

# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(tfidf_train_vectors, y_train)
y_pred = logreg.predict(tfidf_test_vectors)
print(classification_report(y_test, y_pred))
cnf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix (TN, FP, FN, TP):')
print(cnf_matrix)

group_names = ['TN', 'FP', 'FN', 'TP']
group_counts = ["{0:0.0f}".format(value) for value in cnf_matrix.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cnf_matrix, annot=labels, fmt='', cmap='Blues')
#plt.show()

print(f'Jml. Data\t: {tfidf_train_vectors.shape[0]} (80%)')
print(f'Jml. Fitur\t: {tfidf_train_vectors.shape[1]}')
print('Precision\t: {:.2}'.format(precision_score(y_test, y_pred)))
print('Recall\t\t: {:.2}'.format(recall_score(y_test, y_pred)))
print('Accuracy\t: {:.2}'.format(accuracy_score(y_test, y_pred)))
print('F1-Score\t: {:.2}'.format(f1_score(y_test, y_pred)))

filename = 'model-logreg.pickle'
pickle.dump(logreg, open(filename, 'wb'))

# Word2Vec
sentences = [row.split() for row in data['text']]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.save("word2vec.model")

def get_word2vec_vector(text):
    words = text.split()
    vector = np.mean([word2vec_model.wv[word] for word in words if word in word2vec_model.wv], axis=0)
    return vector

data['vector'] = data['text'].apply(lambda x: get_word2vec_vector(x))

X = np.array(data['vector'].tolist())
y = LabelEncoder().fit_transform(data['Sentiment'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(classification_report(y_test, y_pred))
cnf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix (TN, FP, FN, TP):')
print(cnf_matrix)

group_names = ['TN', 'FP', 'FN', 'TP']
group_counts = ["{0:0.0f}".format(value) for value in cnf_matrix.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cnf_matrix, annot=labels, fmt='', cmap='Blues')
#plt.show()

print(f'Jml. Data\t: {X_train.shape[0]} (80%)')
print(f'Jml. Fitur\t: {X_train.shape[1]}')
print('Precision\t: {:.2}'.format(precision_score(y_test, y_pred)))
print('Recall\t\t: {:.2}'.format(recall_score(y_test, y_pred)))
print('Accuracy\t: {:.2}'.format(accuracy_score(y_test, y_pred)))
print('F1-Score\t: {:.2}'.format(f1_score(y_test, y_pred)))

filename = 'model-logreg-word2vec.pickle'
pickle.dump(logreg, open(filename, 'wb'))
print(f'Nama Model\t: {filename}')