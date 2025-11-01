import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score

df = pd.read_csv("D:/MY_AI_AND_ML_PROJECTS/HOUSE PRICE PREDICTION PROJECT/DATASET/Housing.csv")

# Select only object-type columns (likely to contain 'yes'/'no')
obj_cols = df.select_dtypes(include='object').columns
# Apply mapping only to those columns
for col in obj_cols:
    df[col] = df[col].map(lambda x: 1 if x == 'yes' else (0 if x == 'no' else x))

df['furnishingstatus'] = df['furnishingstatus'].map({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})

print(df.info()) # now all are Int64
#features
x = df.drop("price",axis=1)
y = df['price']
#train test split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.1,random_state=42)
#model training
linregmodel = LinearRegression()
linregmodel.fit(x_train,y_train)
#predicitons
y_pred = linregmodel.predict(x_test)
#metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)

print(f"mae - {mae:.2f}")
print(f"mse - {mse:.2f}")
print(f"rmse - {rmse:.2f}")
print(f"r2 score - {r2:.2f}")

#Visualization

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='red', alpha=0.6)

# Line of perfect prediction
max_val = max(max(y_test), max(y_pred)) 
min_val = min(min(y_test), min(y_pred)) 
plt.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', label='Perfect Prediction')

# Labels and title
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.title('Actual vs. Predicted House Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
