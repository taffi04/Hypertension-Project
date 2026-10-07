
#Pharmacy has a tablet where the pharmacist or customer enters.
#1. Age - number 
#2. BMI - number
#Salt intake - perhaps low/moderate/ high.
#stress level - number between 0 - 10.
#Family history of hypertension - Yes/no


print("Welcome to VUT Pharmacy. please enter your details for the hypertension assesssment.")
age = int(input("Please enter your age: "))
bmi = int(input("Enter your BMI: "))
salt_intake = input("Is your salt intake high / moderate / low? ").lower()
stress_level = int(input ("On a scale of 0 - 10, what level is your stress? "))
family_history = input("Does your family have a history of hypertension? (Yes/No): ").lower()

#Then answers go into a Decision Tree

from sklearn.tree import DecisionTreeClassifier

# Training data
X = [
    [65, 30.5, 2, 8, 1],
    [25, 22.0, 0, 3, 0],
    [55, 28.0, 1, 6, 1],
    [35, 24.0, 0, 2, 0],
    [70, 32.0, 2, 9, 1]
]

# 0 = Low Risk
# 1 = Moderate Risk
# 2 = High Risk
y = [2, 0, 1, 0, 2]

# Create Decision Tree
model = DecisionTreeClassifier()

# Train the model
model.fit(X, y)

# Convert Yes/No into numbers
if family_history == "yes":
    family_history_value = 1
else:
    family_history_value = 0

# Convert salt intake into numbers
if salt_intake == "low":
    salt_value = 0
elif salt_intake == "moderate":
    salt_value = 1
else:
    salt_value = 2

patient = [[
    age,
    bmi,
    salt_value,
    stress_level,
    family_history_value
]]

prediction = model.predict(patient)

if prediction[0] == 0:
    print("Hypertension Risk: LOW")
elif prediction[0] == 1:
    print("Hypertension Risk: MODERATE")
else:
    print("Hypertension Risk: HIGH")