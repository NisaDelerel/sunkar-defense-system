from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib

# Dataset yükle
df = pd.read_csv("src/labeled_colorss.csv")

# Tüm renk kanallarını kullan
X = df[['r', 'g', 'b', 'h', 's', 'v', 'l', 'a', 'b_lab']].values
y = df['label'].values

# Etiket encode et
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Eğitim/test ayır
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# SVM modeli (class_weight='balanced' ile)
model = SVC(kernel='linear', class_weight='balanced')
model.fit(X_train, y_train)

# Model ve encoder kaydet
joblib.dump(model, "svm_model.pkl")
joblib.dump(le, "label_encoder.pkl")

# Doğruluk çıktısı
acc = model.score(X_test, y_test)
print(f"Test doğruluk (full features): {acc:.2f}")
