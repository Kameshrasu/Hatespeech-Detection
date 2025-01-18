
# Hate Speech Detection

### **Author:** Kamesh R  


---

## **Introduction**
Hate speech is a growing issue in online spaces, targeting individuals and groups based on factors like race, religion, gender, or sexual orientation.  
The Hate Speech Detection Project addresses this challenge by identifying and categorizing harmful language that incites hate or violence in digital environments.  
The goal is to create a safer and more inclusive online space by detecting and moderating hate speech, promoting healthier digital communities.

---

## **Objective**
To develop an accurate model for detecting hate speech in online text data.

---

## **Methodology**
We leverage **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques to analyze and classify text.

---

## **Key Features**
1. **Data Collection**  
   - **Source:** Curated dataset from Mendeley Data.  
   - **Scope:** Contains over 200,000 samples for comprehensive representation.  
   - **Accessibility:** Open-access dataset to support further research and development.

2. **Balanced Dataset**  
   - **Non-Hate Speech:** 100,452 instances  
   - **Hate Speech:** 98,249 instances  

3. **Data Preprocessing**  
   - **Text Cleaning:** Removes irrelevant elements like punctuation and HTML tags.  
   - **Lowercasing:** Ensures text consistency.  
   - **Tokenization:** Breaks sentences into individual words (tokens).  
   - **Lemmatization & Stemming:** Reduces words to their root forms for analysis.

4. **Feature Extraction**  
   - **Objective:** Transform raw text data into numerical representations for ML models.  
   - **Approach:** Used **TF-IDF (Term Frequency-Inverse Document Frequency)** to highlight words specific to hate speech.

5. **Train-Test Split**  
   - **Training Set:** 80% of the data.  
   - **Testing Set:** 20% of the data.  
   - **Validation:** 20% of the training data.

---

## **Model Building**
Several models were trained and evaluated:  
- **Logistic Regression:** Predicts the probability of a text being hate speech (Accuracy: 81%).  
- **Random Forest:** Combines decision trees for improved accuracy (Accuracy: 81%).  
- **Naive Bayes:** Uses Bayes' theorem to calculate the likelihood (Accuracy: 79%).  
- **Deep Learning:** Advanced neural networks capable of learning complex patterns (Accuracy: 83%).

---

## **Deployment**
The model is deployed as a **Streamlit Web App**, making it accessible and interactive.  
- **Platform:** Streamlit Cloud.  
- **How It Works:** Users enter text, and the system predicts if it contains hate speech.  

---

## **Impact**
The Hate Speech Detection Project contributes to:  
1. **Online Safety:** Protecting individuals and groups from abuse.  
2. **Support for Moderation:** Assisting platforms in managing harmful content.  
3. **Community Well-Being:** Promoting healthier digital spaces.

---






## Live Demo

You can try the application live on Streamlit by clicking the link below:

[Hate Speech Detection App](https://hatespeech-kamesh.streamlit.app/)

