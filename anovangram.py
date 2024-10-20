import pickle
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer  # Ganti TF-IDF dengan N-grams
from sklearn.feature_selection import f_classif  # Ganti Chi-Square dengan ANOVA F-test
from sklearn.feature_selection import SelectKBest
from sklearn import svm
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

# Load dataset
file_dataset = './metopen/dataset_tweet_sentiment_pilkada_DKI_2017.csv'
data = pd.read_csv(file_dataset)

# Data preprocessing
def remove_at_hash(sent):
    return re.sub(r'@|#', r'', sent.lower())

def remove_sites(sent):
    return re.sub(r'http.*', r'', sent.lower())

def remove_punct(sent):
    return ' '.join(re.findall(r'\w+', sent.lower()))

data['text'] = data['Text Tweet'].apply(lambda x: remove_punct(remove_sites(remove_at_hash(x))))

# Encode labels
le = preprocessing.LabelEncoder()
le.fit(data['Sentiment'])
data['label'] = le.transform(data['Sentiment'])

X = data['text']
y = data['label']

# Split data (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

# Ubah teks ke vektor dengan N-grams
# Menggunakan unigrams dan bigrams
bow_vectorizer = CountVectorizer(ngram_range=(1, 2))  # Mengganti BOW dengan N-grams
bow_train_vectors = bow_vectorizer.fit_transform(X_train)
bow_test_vectors = bow_vectorizer.transform(X_test)

# Parameter SVM
pKernel = ['linear', 'rbf']  # kernel SVM
pC = [0.1, 1.0, 10.0]       # nilai C (hyperparameter regulasi)
pGamma = [0.01, 0.1, 1.0]   # nilai gamma untuk kernel non-linear (seperti rbf)
ik = 1                      # indeks untuk kernel (gunakan 'rbf' untuk gamma)
ic = 1                      # indeks untuk nilai C
ig = 1                      # indeks untuk nilai gamma
fs = True  # Seleksi fitur menggunakan ANOVA F-test

print(f'Parameter SVM: Kernel={pKernel[ik]}, C={pC[ic]}, Gamma={pGamma[ig]}')

# Seleksi fitur dengan ANOVA F-test
if fs:
    fs_label = "ANOVA"
    anova = SelectKBest(f_classif, k=900)  # Menggunakan ANOVA F-test
    bow_train_vectors = anova.fit_transform(bow_train_vectors, y_train)
    bow_test_vectors = anova.transform(bow_test_vectors)
else:
    fs_label = "None"

print(f'Seleksi Fitur SVM: {fs_label}')

# Train SVM classifier with gamma
svm_classifier = svm.SVC(kernel=pKernel[ik], C=pC[ic], gamma=pGamma[ig])
svm_classifier.fit(bow_train_vectors, y_train)

# Predict and evaluate
y_pred = svm_classifier.predict(bow_test_vectors)
print(classification_report(y_test, y_pred))

cnf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix (TN, FP, FN, TP):')
print(cnf_matrix)

# Confusion matrix visualization
group_names = ['TN', 'FP', 'FN', 'TP']
group_counts = ["{0:0.0f}".format(value) for value in cnf_matrix.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cnf_matrix, annot=labels, fmt='', cmap='Blues')

#plt.show()

# Print metrics
print(f'Sel. Fitur\t: {fs_label}')
print(f'Param. SVM\t: Kernel={pKernel[ik]}, C={pC[ic]}, Gamma={pGamma[ig]}')
print(f'Jml. Data\t: {bow_train_vectors.shape[0]} (80%)')
print(f'Jml. Fitur\t: {bow_train_vectors.shape[1]}')
print('Precision\t: {:.2}'.format(precision_score(y_test, y_pred)))
print('Recall\t\t: {:.2}'.format(recall_score(y_test, y_pred)))
print('Accuracy\t: {:.2}'.format(accuracy_score(y_test, y_pred)))
print('F1-Score\t: {:.2}'.format(f1_score(y_test, y_pred)))

# Simpan model
filename = f'model-svm-{fs_label}-{pKernel[ik]}-{pC[ic]}-{pGamma[ig]}.pickle'
pickle.dump(svm_classifier, open(filename, 'wb'))

# Simpan model dengan vectorizer dan ANOVA
vectorizer = bow_vectorizer
clf = svm_classifier
with open(filename, 'wb') as fout:
    if fs:
        pickle.dump((vectorizer, anova, clf), fout)
    else:
        pickle.dump((vectorizer, clf), fout)

print(f'Nama Model\t: {filename}')