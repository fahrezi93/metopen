import matplotlib.pyplot as plt


# data anova + ngram
percobaan_anova_ngram = [
    '1 (None, Linear, 0.1, Scale)', '2 (None, Linear, 1.0, Auto)', '3 (None, Linear, 10, 0.1)',
    '4 (None, RBF, 0.1, Scale)', '5 (None, RBF, 1.0, Auto)', '6 (None, RBF, 10, 0.1)',
    '7 (Fitur, Linear, 0.1, Scale)', '8 (Fitur, Linear, 1.0, Auto)',
    '9 (Fitur, Linear, 10, 0.1)', '10 (Fitur, RBF, 0.1, Scale)',
    '11 (Fitur, RBF, 1.0, Auto)', '12 (Fitur, RBF, 10, 0.1)'
]

# Akurasi dan F-Measure
akurasi_anova_ngram = [0.74, 0.76, 0.72, 0.55, 0.51, 0.52, 0.72, 0.71, 0.68, 0.61, 0.53, 0.64]

# Data anova bow
percobaan_anova_bow = [
    '1 (None, Linear, 0.1, Scale)', '2 (None, Linear, 1.0, Auto)', '3 (None, Linear, 10, 0.1)',
    '4 (None, RBF, 0.1, Scale)', '5 (None, RBF, 1.0, Auto)', '6 (None, RBF, 10, 0.1)',
    '7 (Fitur, Linear, 0.1, Scale)', '8 (Fitur, Linear, 1.0, Auto)',
    '9 (Fitur, Linear, 10, 0.1)', '10 (Fitur, RBF, 0.1, Scale)',
    '11 (Fitur, RBF, 1.0, Auto)', '12 (Fitur, RBF, 10, 0.1)'
]

# Akurasi dan F-Measure
akurasi_anova_bow = [0.72, 0.74, 0.73, 0.63, 0.51, 0.71, 0.72, 0.71, 0.73, 0.64, 0.53, 0.73]

# Plotting data
plt.figure(figsize=(12, 6))

plt.plot(percobaan_anova_ngram, akurasi_anova_ngram, marker='o', label='N-gram + Anova')
plt.plot(percobaan_anova_bow, akurasi_anova_bow, marker='o', label='BoW + Anova')

# Menambahkan judul dan label
plt.title('Perbandingan Metode Ekstraksi Fitur dengan N-gram dan BoW')
plt.xlabel('Percobaan (Fiture Selection, Kernel, C)')
plt.ylabel('Akurasi')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)

# Menambahkan grid dan legenda
plt.grid(True)
plt.legend()

# Menampilkan plot
plt.tight_layout()
plt.show()