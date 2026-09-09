import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('gia_nha.csv')

X = df[['Dien_Tich_m2', 'So_Phong_Ngu', 'Khoang_Cach_Trung_Tam_km']]
y = df['Gia_Ty_VND']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- KẾT QUẢ ĐÁNH GIÁ ---")
print(f"Sai số toàn phương trung bình (MSE): {mse:.2f}")
print(f"Hệ số xác định (R2 Score): {r2:.2f}")

print(f"\n--- HỆ SỐ CỦA MÔ HÌNH ---")
for feature, coef in zip(X.columns, model.coef_):
    print(f"- {feature}: {coef:.3f}")
print(f"- Hệ số tự do (Intercept): {model.intercept_:.3f}")
print("Đây là phiên bản chạy trên nhánh thử nghiệm!")