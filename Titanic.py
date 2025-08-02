import pandas as pd

# Load the uploaded Titanic dataset
file_path = 'titanic.csv'
titanic_df = pd.read_csv(file_path)

# Display basic info and first few rows for initial exploration
titanic_df.info(), titanic_df.head()

# Start data preprocessing
# First, how many missing values we have in each column
missing_values = titanic_df.isnull().sum()
missing_values

# Drop 'Cabin' due to too many missing values
titanic_df.drop(columns=['Cabin'], inplace=True)


# Fill missing 'Age' and 'Fare' with their respective median values
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())
titanic_df['Fare'] = titanic_df['Fare'].fillna(titanic_df['Fare'].median())

# Confirm that all missing values are handled
titanic_df.isnull().sum()

# Convert 'Sex' to binary
titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked' (C, Q, S) — drop_first avoids multicollinearity
embarked_dummies = pd.get_dummies(titanic_df['Embarked'], prefix='Embarked', drop_first=True)

# Add the one-hot encoded columns
titanic_df = pd.concat([titanic_df, embarked_dummies], axis=1)

# Drop the original 'Embarked' column
titanic_df.drop(columns=['Embarked'], inplace=True)

#Step 1: Feature Selection
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X = titanic_df[features]      # input features
y = titanic_df['Survived']    # target variable

#Step 2: Split the Data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Evaluate the Model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print("Average CV Accuracy:", scores.mean())





