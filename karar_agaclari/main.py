import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

veri = pd.read_csv("breast_cancer.csv")# veri seti yükleme
print(veri.isnull().sum()) # eksik değer kontrolü
X = veri.drop(columns=["Class"])#veri özelliklri ve hedef  degisken
y = veri["Class"]
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=42)#eğitim ve test setlerine ayırma
model = DecisionTreeClassifier(random_state=42)
model.fit(X_egitim, y_egitim)

tahminler = model.predict(X_test)

dogruluk = accuracy_score(y_test, tahminler)
print("Doğruluk (Accuracy):", dogruluk)
f1_skoru = f1_score(y_test, tahminler, average='weighted')
print("f1 Skoru:", f1_skoru)
hassasiyet = precision_score(y_test, tahminler, average='weighted')
print("Hassasiyet:", hassasiyet)
geri_ccagirma = recall_score(y_test, tahminler, average='weighted')
print("Geri Çağırma:", geri_ccagirma)

karisiklik_matrisi = confusion_matrix(y_test, tahminler)#karışıklık matrisi hesaplama
plt.figure(figsize=(8, 6))
sns.heatmap(karisiklik_matrisi, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.title('Karışıklık Matrisi')
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns,class_names=['2',"4"], filled=True)
plt.title('Karar Ağacı görselleştirmesi')
plt.show()