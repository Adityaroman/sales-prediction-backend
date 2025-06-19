import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import joblib

df = pd.read_csv('data/sales_data.csv')
df.dropna(inplace=True)
df['Amount'] = df['Amount'].astype(int)
df['Age_Group'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 70], labels=['18-25', '26-35', '36-45', '46-55', '56-70'])

le_gender = LabelEncoder()
le_marital = LabelEncoder()
le_state = LabelEncoder()
le_category = LabelEncoder()
le_age_group = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Marital_Status'] = le_marital.fit_transform(df['Marital_Status'])
df['State'] = le_state.fit_transform(df['State'])
df['Product_Category'] = le_category.fit_transform(df['Product_Category'])
df['Age_Group'] = le_age_group.fit_transform(df['Age_Group'])

joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_marital, 'le_marital.pkl')
joblib.dump(le_state, 'le_state.pkl')
joblib.dump(le_category, 'le_category.pkl')
joblib.dump(le_age_group, 'le_age_group.pkl')

X = df[['Age', 'Gender', 'Marital_Status', 'State', 'Product_Category', 'Age_Group', 'Orders']]
y = df['Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, r2_score
y_pred = model.predict(X_test)
print(f'MAE: {mean_absolute_error(y_test, y_pred):.2f}')
print(f'R-squared: {r2_score(y_test, y_pred):.2f}')

joblib.dump(model, 'sales_model.pkl')
print("Model and encoders saved successfully")
