import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Load Data
file_path = './dataset.csv'
data = pd.read_csv(file_path)

#Check for Missing Values
missing_values = data.isnull().sum()
#Check for Duplicate Values
duplicate_values = data.duplicated().sum()
#Check for Outliers (using Boxplot)
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[['Age', 'App Sessions', 'Distance Travelled (km)', 'Calories Burned']])
plt.title('Boxplot for Outlier Detection')
plt.show()

#Gender Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=data)
plt.title('Gender Distribution')
plt.show()
#Age Distribution
plt.figure(figsize=(6, 4))
sns.histplot(data['Age'], bins=10, kde=True)
plt.title('Age Distribution')
plt.show()
#Activity Level Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Activity Level', data=data)
plt.title('Activity Level Distribution')
plt.show()
#Location Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Location', data=data)
plt.title('Location Distribution')
plt.show()
#App Sessions Distribution
plt.figure(figsize=(6, 4))
sns.histplot(data['App Sessions'], bins=10, kde=True)
plt.title('App Sessions Distribution')
plt.show()
#Distance Travelled Distribution
plt.figure(figsize=(6, 4))
sns.histplot(data['Distance Travelled (km)'], bins=10, kde=True)
plt.title('Distance Travelled Distribution')
plt.show()
#Calories Burned Distribution
plt.figure(figsize=(6, 4))
sns.histplot(data['Calories Burned'], bins=10, kde=True)
plt.title('Calories Burned Distribution')
plt.show()
#Correlation between User Age and App Session Count
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Age', y='App Sessions', data=data)
plt.title('Age vs App Sessions')
plt.show()
#Relationship between User Gender and Calories Burned
plt.figure(figsize=(6, 4))
sns.boxplot(x='Gender', y='Calories Burned', data=data)
plt.title('Gender vs Calories Burned')
plt.show()

#Relationship between Activity Level, Travel Distance, and Calories Burned
plt.figure(figsize=(6, 4))
sns.boxplot(x='Activity Level', y='Distance Travelled (km)', data=data)
plt.title('Activity Level vs Distance Travelled')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='Activity Level', y='Distance Travelled (km)', data=data)
plt.title('Activity Level vs Distance Travelled')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='Activity Level', y='Calories Burned', data=data)
plt.title('Activity Level vs Calories Burned')
plt.show()
#Differences in App Session Counts Among Users from Different Geographical Locations
plt.figure(figsize=(6, 4))
sns.boxplot(x='Location', y='App Sessions', data=data)
plt.title('Location vs App Sessions')
plt.show()

#Model Building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#Translation: Regression Model (Predicting Calorie Burn)
X = data[['Age', 'App Sessions', 'Distance Travelled (km)']]
y = data['Calories Burned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
#Evaluation of the Model
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Calories Burned')
plt.ylabel('Predicted Calories Burned')
plt.title('Actual vs Predicted Calories Burned')
plt.show()
#Classification Model (Predicting User Activity)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
le = LabelEncoder()
data['Activity Level'] = le.fit_transform(data['Activity Level'])
X = data[['Age', 'App Sessions', 'Distance Travelled (km)', 'Calories Burned']]
y = data['Activity Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()