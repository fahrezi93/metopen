import pickle
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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

# Doc2Vec model
documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(X_train)]
doc2vec_model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)
doc2vec_model.save("doc2vec.model")

# Convert text to vectors
def get_doc2vec_vector(text):
    return doc2vec_model.infer_vector(text.split())

X_train_vectors = np.array([get_doc2vec_vector(text) for text in X_train])
X_test_vectors = np.array([get_doc2vec_vector(text) for text in X_test])

# Mutual Information for feature selection
mi = mutual_info_classif(X_train_vectors, y_train)
indices = np.argsort(mi)[-900:]  # Select top 900 features

X_train_vectors = X_train_vectors[:, indices]
X_test_vectors = X_test_vectors[:, indices]

# SVM with gamma
pKernel = ['linear', 'rbf']
pC = [0.1, 1.0, 10.0]
pGamma = [0.01, 0.1, 1.0]
ik = 0
ic = 0
ig = 0

svm_classifier = svm.SVC(kernel=pKernel[ik], C=pC[ic], gamma=pGamma[ig])
svm_classifier.fit(X_train_vectors, y_train)
y_pred = svm_classifier.predict(X_test_vectors)
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

print(f'Sel. Fitur\t: Mutual Information')
print(f'Ekstraksi\t: Doc2Vec')
print(f'Param. SVM\t: Kernel={pKernel[ik]}, C={pC[ic]}, Gamma={pGamma[ig]}')
print(f'Jml. Data\t: {X_train_vectors.shape[0]} (80%)')
print(f'Jml. Fitur\t: {X_train_vectors.shape[1]}')
print('Precision\t: {:.2}'.format(precision_score(y_test, y_pred)))
print('Recall\t\t: {:.2}'.format(recall_score(y_test, y_pred)))
print('Accuracy\t: {:.2}'.format(accuracy_score(y_test, y_pred)))
print('F1-Score\t: {:.2}'.format(f1_score(y_test, y_pred)))

filename = f'model-svm-MI-Doc2Vec-{pKernel[ik]}-{pC[ic]}-{pGamma[ig]}.pickle'
pickle.dump(svm_classifier, open(filename, 'wb'))

# Random Forest Classifier
rf_classifier.fit(X_train_vectors, y_train)
y_pred = rf_classifier.predict(X_test_vectors)
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

print(f'Jml. Data\t: {X_train_vectors.shape[0]} (80%)')
print(f'Jml. Fitur\t: {X_train_vectors.shape[1]}')
print('Precision\t: {:.2}'.format(precision_score(y_test, y_pred)))
print('Recall\t\t: {:.2}'.format(recall_score(y_test, y_pred)))
print('Accuracy\t: {:.2}'.format(accuracy_score(y_test, y_pred)))
print('F1-Score\t: {:.2}'.format(f1_score(y_test, y_pred)))

filename = 'model-rf-MI-Doc2Vec.pickle'
pickle.dump(rf_classifier, open(filename, 'wb'))
print(f'Nama Model\t: {filename}')