# ========================================================
# Project: Healthcare Diagnostic Chatbot
# Author: Rudra Pratap Pandey
# Tech: Decision Tree Classifier & Localized Input Handler
# ========================================================

import csv
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree


class HealthcareEngine:
    def __init__(self):
        # Paths for datasets
        self.data_dir = Path("Data")
        self.master_dir = Path("MasterData")

        # Localized response sets (Supports Hinglish/Hindi)
        self.yes_words = {"yes", "y", "haa", "ha", "haan", "han", "hao", "haanji"}
        self.no_words = {"no", "n", "naa", "na", "nahi", "nahin", "nhi"}

        # Dictionaries for metadata
        self.severity_dict = {}
        self.description_dict = {}
        self.precaution_dict = {}

        # ML Model and feature stores
        self.clf = DecisionTreeClassifier()
        self.feature_cols = []
        self.training_df = pd.DataFrame()
        self.reduced_data = pd.DataFrame()

    def load_datasets(self):
        # Reading symptom severity ratings
        severity_file = self.master_dir / "Symptom_severity.csv"
        if severity_file.exists():
            with open(severity_file, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        try:
                            self.severity_dict[row[0].strip()] = int(row[1])
                        except ValueError:
                            continue

        # Reading disease descriptions
        desc_file = self.master_dir / "symptom_Description.csv"
        if desc_file.exists():
            with open(desc_file, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        self.description_dict[row[0].strip()] = row[1].strip()

        # Reading precautionary measures
        precaution_file = self.master_dir / "symptom_precaution.csv"
        if precaution_file.exists():
            with open(precaution_file, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 5:
                        self.precaution_dict[row[0].strip()] = [x.strip() for x in row[1:5]]

    def train_model(self):
        # Training the Decision Tree Classifier
        train_path = self.data_dir / "Training.csv"
        self.training_df = pd.read_csv(train_path)

        self.feature_cols = list(self.training_df.columns[:-1])
        X = self.training_df[self.feature_cols]
        y = self.training_df["prognosis"]

        self.reduced_data = self.training_df.groupby(self.training_df["prognosis"]).mean()
        self.clf.fit(X, y)

    def validate_input(self, prompt_text):
        # Handling input validation for Hinglish responses
        print(prompt_text, end="")
        while True:
            user_inp = input("").strip().lower()
            if user_inp in self.yes_words:
                return True
            if user_inp in self.no_words:
                return False
            print("provide proper answers i.e. (yes/no) : ", end="")

    def sec_predict(self, symptoms_exp):
        # Secondary disease evaluation using symptom vector matching
        input_vector = np.zeros(len(self.feature_cols))
        for item in symptoms_exp:
            if item in self.feature_cols:
                input_vector[self.feature_cols.index(item)] = 1

        df_input = pd.DataFrame([input_vector], columns=self.feature_cols)
        prediction = self.clf.predict(df_input)
        return list(prediction)

    def run_diagnostic(self):
        print("--------------------------------HealthCare ChatBot--------------------------------")
        user_name = input("Your Name? \n- ")
        print(f"Hello, {user_name}")

        primary_symptom = input("Enter the symptom you are experiencing \n- ").strip()
        num_days = input("Okay. From how many days ? : ").strip()

        tree_ = self.clf.tree_
        feature_names = [
            self.feature_cols[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        symptoms_exp = []

        def recurse(node, depth):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names[node]
                val = 1 if name == primary_symptom else 0

                if val <= tree_.threshold[node]:
                    recurse(tree_.children_left[node], depth + 1)
                else:
                    symptoms_exp.append(name)
                    recurse(tree_.children_right[node], depth + 1)
            else:
                # Leaf node: primary prediction
                disease_values = tree_.value[node]
                predicted_index = np.argmax(disease_values)
                present_disease = [self.clf.classes_[predicted_index]]

                red_cols = self.reduced_data.columns
                raw_symptoms = red_cols[self.reduced_data.loc[present_disease].values[0] > 0]
                
                # Filter out the primary symptom to prevent duplicate prompts
                symptoms_given = [sym for sym in raw_symptoms if sym != primary_symptom]

                # Ask follow-up questions only if secondary symptoms exist
                if len(symptoms_given) > 0:
                    print("Are you experiencing any ")
                    for syms in list(symptoms_given):
                        if self.validate_input(f"{syms} ? : "):
                            symptoms_exp.append(syms)

                second_prediction = self.sec_predict(symptoms_exp)

                if present_disease[0] == second_prediction[0]:
                    print("\nYou may have ", present_disease[0])
                    print(self.description_dict.get(present_disease[0], ""))
                else:
                    print("\nYou may have ", present_disease[0], "or ", second_prediction[0])
                    print(self.description_dict.get(present_disease[0], ""))
                    print(self.description_dict.get(second_prediction[0], ""))

                precaution_list = self.precaution_dict.get(present_disease[0], [])
                if precaution_list:
                    print("\nTake following measures : ")
                    for i, j in enumerate(precaution_list):
                        print(i + 1, ")", j)

        recurse(0, 1)

    def start(self):
        self.load_datasets()
        self.train_model()
        self.run_diagnostic()


if __name__ == "__main__":
    bot = HealthcareEngine()
    bot.start()