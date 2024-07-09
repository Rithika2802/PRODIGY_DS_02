# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
titanic_data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Display the first few rows of the dataset
print(titanic_data.head())

# Check for missing values
print(titanic_data.isnull().sum())

# Data Cleaning

# Fill missing values for 'Age' with median age
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)

# Fill missing values for 'Embarked' with mode
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' column due to too many missing values
titanic_data.drop('Cabin', axis=1, inplace=True)

# EDA

# Visualize survival count
sns.countplot(x='Survived', data=titanic_data)
plt.title('Survival Count')
plt.show()

# Visualize survival count by sex
sns.countplot(x='Survived', hue='Sex', data=titanic_data)
plt.title('Survival Count by Sex')
plt.show()

# Visualize survival count by passenger class
sns.countplot(x='Survived', hue='Pclass', data=titanic_data)
plt.title('Survival Count by Passenger Class')
plt.show()

# Visualize age distribution
sns.histplot(titanic_data['Age'], bins=20)
plt.title('Age Distribution')
plt.show()

# Visualize fare distribution
sns.histplot(titanic_data['Fare'], bins=20)
plt.title('Fare Distribution')
plt.show()

# Visualize survival count by age
sns.histplot(x='Age', hue='Survived', data=titanic_data, bins=20, multiple='stack')
plt.title('Survival Count by Age')
plt.show()

# Visualize survival count by fare
sns.histplot(x='Fare', hue='Survived', data=titanic_data, bins=20, multiple='stack')
plt.title('Survival Count by Fare')
plt.show()

# Visualize survival count by embarked port
sns.countplot(x='Survived', hue='Embarked', data=titanic_data)
plt.title('Survival Count by Embarked Port')
plt.show()

# Correlation matrix - Select only numerical columns
numerical_data = titanic_data.select_dtypes(include=[np.number]) # Select numerical columns only
correlation_matrix = numerical_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
