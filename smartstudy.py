# smart study time recommender
# made for AI/ML project (simple version)

import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------------------------
# creating small dataset manually
# (just for demo purpose)
# ---------------------------
data = {
    "subject": ["Math", "Physics", "Chemistry", "English", "Biology",
                "Math", "Physics", "Chemistry"],
    "difficulty": [5, 4, 3, 2, 4, 5, 4, 3],
    "days_left": [10, 8, 7, 5, 6, 5, 3, 4],
    "importance": [5, 4, 3, 2, 4, 5, 4, 3],
    "study_hours": [3, 2.5, 2, 1.5, 2.5, 4, 3, 2.5]
}

df = pd.DataFrame(data)

# converting subject names into numbers (easy way)
df["subject"] = df["subject"].astype("category").cat.codes

# inputs and output
X = df[["subject", "difficulty", "days_left", "importance"]]
y = df["study_hours"]

# making model
model = LinearRegression()

# training model
model.fit(X, y)

print("model is ready now\n")

# ---------------------------
# taking input from user
# ---------------------------
subjects = ["Math", "Physics", "Chemistry", "English", "Biology"]

print("Subjects available:")
for i in range(len(subjects)):
    print(i, subjects[i])

# user inputs
sub = int(input("enter subject number: "))
diff = int(input("difficulty (1-5): "))
days = int(input("days left for exam: "))
imp = int(input("importance (1-5): "))

# putting input in proper format
user_data = pd.DataFrame([[sub, diff, days, imp]],
                         columns=["subject", "difficulty", "days_left", "importance"])

# prediction
pred = model.predict(user_data)

# small extra logic (my idea)
if days <= 3:
    pred[0] = pred[0] + 1
    print("\nexam is very close, so try to study more")

# final output
print("\nyou should study approx", round(pred[0], 2), "hours daily")
print("all the best for your exams 👍")
