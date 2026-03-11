import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df=pd.read_csv('train.csv')

## filling empty cells of Age
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna("S")

# Encode
le_sex = LabelEncoder()
le_emb = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex"])
df["Embarked"] = le_emb.fit_transform(df["Embarked"])

features=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
x=df[features]
y=df["Survived"]

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

pred=model.predict(x_val)
acc=accuracy_score(y_val,pred)

test_df = pd.read_csv("test.csv")

test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median())
test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())
test_df["Embarked"] = test_df["Embarked"].fillna("S")

test_df["Sex"] = le_sex.transform(test_df["Sex"])
test_df["Embarked"] = le_emb.transform(test_df["Embarked"])

X_test = test_df[features]
predictions = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": predictions
})

submission.to_csv("submission.csv", index=False)
