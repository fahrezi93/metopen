import pickle
filename = 'model-svm-None-linear-1.0.pickle'
fs_label = filename[10:13]; # Chi
text = ['pilkada akan dimenangkan oleh anies']
with open(filename, 'rb') as fin:
 if fs_label == "Chi":
  vectorizer, ch2, clf = pickle.load(fin)
  tfidf_text_vectors = vectorizer.transform(text)
  tfidf_text_vectors = ch2.transform(tfidf_text_vectors)
 else:
  vectorizer, clf = pickle.load(fin)
  tfidf_text_vectors = vectorizer.transform(text)
y_pred = clf.predict(tfidf_text_vectors)
print(y_pred)
if y_pred:
 sentimen_svm = 'Tweet positif'
else:
 sentimen_svm = 'Tweet negatif'
print('Teks\t\t: ', text[0])
print('Sentimen\t: ', sentimen_svm) 