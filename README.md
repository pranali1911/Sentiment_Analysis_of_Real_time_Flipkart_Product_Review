# Sentiment_Analysis_of_Real_time_Flipkart_Product_Reviews
Awesome — here’s your **FULL, FINAL, ready-to-paste `README.md`** for your GitHub project.
This is clean, professional, internship-ready, and matches your MLflow + Streamlit + AWS setup 👌
Just copy everything below into a file named **`README.md`** in your repo.

---

# 📊 Sentiment Analysis of Real-time Flipkart Product Reviews

**(End-to-End NLP + MLOps with MLflow | Internship Project @ Innomatics Research Labs)**

This project is an end-to-end **Sentiment Analysis system** built on real Flipkart product reviews (YONEX Nylon Shuttle).
It classifies customer reviews as **Positive** or **Negative** and helps identify **customer pain points** from negative feedback.
The project integrates **MLflow** for experiment tracking, hyperparameter tuning, model management, and reproducibility as part of an **MLOps workflow**.
A **Streamlit web app** is built for real-time predictions and deployed on **AWS EC2**.

---

## 🚀 Project Highlights

* 🔹 Real-world Flipkart reviews dataset (provided)
* 🔹 End-to-end NLP pipeline: cleaning → feature extraction → modeling → deployment
* 🔹 Multiple ML models trained & compared using **F1-score**
* 🔹 **MLflow** for experiment tracking, metrics, parameters, artifacts, and model logging
* 🔹 Hyperparameter tuning using **Optuna** (logged to MLflow)
* 🔹 Interactive **Streamlit web app** for real-time sentiment prediction
* 🔹 Deployed on **AWS EC2**
* 🔹 Completed as part of **Internship at Innomatics Research Labs**

---

## 🧠 Problem Statement

Classify customer reviews into **Positive** or **Negative** sentiment and analyze negative reviews to understand customer pain points.
This helps businesses improve product quality and customer experience based on real feedback.

---

## 📂 Dataset

* Product: **YONEX MAVIS 350 Nylon Shuttle**
* Total Reviews: **8,500+**
* Features include:

  * Reviewer Name
  * Rating
  * Review Title
  * Review Text
  * Place & Date of Review
  * Upvotes / Downvotes

> ⚠️ Dataset was provided as part of the project instructions. Data was not scraped manually.

---

## 🔄 Project Workflow

1. **Data Loading & EDA**
2. **Text Preprocessing**

   * Lowercasing
   * Removing special characters
   * Stopword removal
   * Lemmatization / Stemming
3. **Feature Engineering**

   * Bag of Words (BoW)
   * TF-IDF
4. **Model Training**

   * Logistic Regression
   * Linear SVM
   * SVM (RBF)
   * Random Forest
   * Multinomial Naive Bayes
5. **Evaluation**

   * Primary metric: **F1-score**
6. **MLOps with MLflow**

   * Experiment tracking
   * Hyperparameter tuning (Optuna)
   * Model logging
7. **Deployment**

   * Streamlit web app
   * AWS EC2 hosting

---

## 🛠️ Tech Stack

* **Programming:** Python
* **Libraries:** Pandas, NumPy, NLTK, Scikit-learn
* **NLP:** TF-IDF, Bag of Words
* **MLOps:** MLflow, Optuna
* **Web App:** Streamlit
* **Deployment:** AWS EC2
* **Version Control:** Git & GitHub

---

## ⚙️ How to Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/pranali1911/Sentiment_Analysis_of_Real_time_Flipkart_Product_Review.git
cd Sentiment_Analysis_of_Real_time_Flipkart_Product_Review
```

### 2️⃣ Create Virtual Environment (Optional but recommended)

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux / Mac
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run MLflow UI

```bash
mlflow ui
```

Open in browser:
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 5️⃣ Run Training with MLflow

```bash
python MLFLOW_sentiment_analysis.py
```

### 6️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 📊 MLflow Tracking

MLflow is used to track:

* Parameters (vectorizer + model hyperparameters)
* Metrics (CV F1, Train/Test Accuracy, Precision, Recall, F1)
* Artifacts (confusion matrix, classification report)
* Trained models

This helps in:

* Comparing different models
* Selecting the best model based on **F1-score**
* Reproducibility of experiments

---

## 📸 Project Screenshots


### 🔹 MLflow – Experiments Dashboard

<img width="1903" height="959" alt="Screenshot 2026-02-14 233440" src="https://github.com/user-attachments/assets/68f9e45b-fd6c-4e04-8eee-842b57ed84e8" />


### 🔹 MLflow – Compare Runs (Metrics & Parameters)
<img width="1857" height="827" alt="Screenshot 2026-02-14 234431" src="https://github.com/user-attachments/assets/37ad6ac4-790b-463f-9054-0bed7257e7bf" />


### 🔹 MLflow – Parallel Coordinates Plot

<img width="1817" height="778" alt="Screenshot 2026-02-14 235454" src="https://github.com/user-attachments/assets/4a8dc6a6-6693-4cdf-ad18-49226ac5a563" />


### 🔹 Streamlit Web App – Sentiment Prediction UI

<img width="1890" height="941" alt="Screenshot 2026-02-09 223222" src="https://github.com/user-attachments/assets/7e9673dc-535e-4f56-b9d7-adb50eb58150" />


<img width="1905" height="964" alt="Screenshot 2026-02-09 223908" src="https://github.com/user-attachments/assets/721ecff0-5dc0-45b9-a42d-80327bc8971d" />



---

## 🌐 Live Demo

🔗 **Streamlit App (AWS EC2):**
[http://13.235.42.40:8501/](http://13.235.42.40:8501/)

---

## 📌 Results & Insights

* Most Flipkart reviews are **positive**, showing overall customer satisfaction.
* **Negative reviews are longer** and contain detailed complaints.
* **TF-IDF + Linear Models (Logistic Regression / Linear SVM)** performed best.
* MLflow made it easy to **compare experiments and select the best model**.

---

## 🙌 Acknowledgements

This project was completed as part of my internship at **Innomatics Research Labs**.
Thanks to **MLflow** for enabling proper experiment tracking and model management.

---

## 📬 Feedback

Feedback and suggestions are welcome!
Feel free to open an issue or connect with me on LinkedIn 😊

---

## 📄 License

This project is for educational and portfolio purposes.

---

If you want, send me your **GitHub repo link** and I’ll quickly review this README on your actual repo and suggest final tweaks before submission 🚀
