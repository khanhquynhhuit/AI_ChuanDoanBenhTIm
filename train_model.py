import pandas as pd
from sklearn.tree import DecisionTreeClassifier # <--- Đổi thư viện
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. Đọc dữ liệu
df = pd.read_csv('heart.csv')

# 2. Chia dữ liệu
X = df.drop('target', axis=1)
y = df['target']

# 3. Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Khởi tạo Decision Tree
# criterion='gini': Tiêu chuẩn đo độ vẩn đục (phổ biến nhất)
# random_state=42: Để kết quả cố định mỗi lần chạy
model = DecisionTreeClassifier(criterion='gini', random_state=42) 

# 5. Huấn luyện
model.fit(X_train, y_train)

# 6. Đánh giá
y_pred = model.predict(X_test)
print(f"Độ chính xác của Decision Tree: {accuracy_score(y_test, y_pred):.2f}")

# 7. Lưu model
joblib.dump(model, 'heart_model.pkl')
print("Đã lưu mô hình Decision Tree thành công!")