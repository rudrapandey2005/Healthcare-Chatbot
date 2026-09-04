# 🩺 Healthcare Diagnostic Engine

An interactive, Object-Oriented Machine Learning Healthcare Chatbot written in Python. Built using a Decision Tree Classifier, this engine predicts potential medical conditions through dynamic symptom traversal, supports localized Hinglish/Hindi response validation, and provides detailed disease descriptions alongside step-by-step precautionary measures.

---

## 👨‍💻 Author
**Rudra Pratap Pandey**  
*GitHub:* [rudrapandey2005](https://github.com/rudrapandey2005)

---

## ✨ Key Features

* **Object-Oriented Architecture:** Encapsulated within a modular `HealthcareEngine` class structure for improved maintainability and clean execution flow.
* **Machine Learning Diagnostic Core:** Utilizes `DecisionTreeClassifier` trained on medical symptom datasets, supported by secondary symptom-vector matching for comprehensive disease evaluation.
* **Localized Input Validation:** Built-in handler accepting both standard English (`yes`/`no`) and Hinglish/Hindi responses (`haa`, `naa`, `nahi`, `haanji`, `nhi`).
* **Dynamic Tree Traversal:** Iterates through symptom nodes and prompts follow-up questions only when valid secondary symptoms are identified.
* **Actionable Medical Insights:** Automatically maps predicted diagnoses to detailed disease descriptions and actionable 4-step precautionary steps.

---

## 🛠️ Tech Stack

* **Language:** Python 3.8+
* **Machine Learning:** Scikit-learn (`DecisionTreeClassifier`)
* **Data Processing:** Pandas, NumPy
* **File Management:** Pathlib, CSV

---

## 📁 Repository Structure

```text
Healthcare-Chatbot/
├── Data/
│   ├── Training.csv
│   └── Testing.csv
├── MasterData/
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   └── Symptom_severity.csv
├── chat_bot.py
├── requirements.txt
└── README.md