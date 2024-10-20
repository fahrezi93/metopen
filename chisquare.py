import pickle
import pandas as pd
import re
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.decomposition import PCA

# Load dataset
file_dataset = 'dataset_tweet_sentiment_pilkada_DKI_2017.csv'
data = pd.read_csv(file_dataset)

# Cleaning dataset
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

# Convert text to vectors with CountVectorizer
count_vectorizer = CountVectorizer()
count_train_vectors = count_vectorizer.fit_transform(X_train)
count_test_vectors = count_vectorizer.transform(X_test)

# Define the parameter lists
pKernel = ['linear', 'rbf']
pC = [0.1, 1.0, 10.0]
pGamma = ['auto', 'scale', 0.1]
feature_selection_methods = [None, 'PCA']

# Loop through all combinations of parameters
for ik in range(len(pKernel)):
    for ic in range(len(pC)):
        for ig in range(len(pGamma)):
            for ifs in range(len(feature_selection_methods)):

                # Print the chosen parameters
                print(f'\nParameter SVM: Kernel={pKernel[ik]}, C={pC[ic]}, Gamma={pGamma[ig]}, Feature Selection={feature_selection_methods[ifs]}')

                # Apply feature selection if chosen
                if feature_selection_methods[ifs] == 'PCA':
                    # Initialize PCA with the number of components limited by the minimum between the desired and the available features/samples limit
                    n_components = min(1500, count_train_vectors.shape[0], count_train_vectors.shape[1])
                    pca = PCA(n_components=n_components)

                    # Fit and transform the training set
                    count_train_vectors_fs = pca.fit_transform(count_train_vectors.toarray())

                    # Transform the test set
                    count_test_vectors_fs = pca.transform(count_test_vectors.toarray())
                else:
                    count_train_vectors_fs = count_train_vectors
                    count_test_vectors_fs = count_test_vectors
                    pca = None

                # Set the kernel type, C value, and gamma
                kernel = pKernel[ik]
                C = pC[ic]
                gamma = pGamma[ig]

                # Initialize the SVM classifier with selected parameters
                svm_classifier = svm.SVC(kernel=kernel, C=C, gamma=gamma)

                # Train the SVM model
                svm_classifier.fit(count_train_vectors_fs, y_train)
                y_pred = svm_classifier.predict(count_test_vectors_fs)

                # Calculate performance metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')

                # Output the accuracy and F1 score
                print(f'Accuracy: {accuracy}')
                print(f'F1-Score: {f1}')