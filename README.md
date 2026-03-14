👕 Women's Clothing Reviews Sentiment Analysis


An end-to-end Natural Language Processing (NLP) project that analyzes customer reviews from a women's clothing e-commerce platform and predicts sentiment using TF-IDF + Logistic Regression.

The project includes:

📊 Data Exploration

🤖 Machine Learning Model

🌐 FastAPI Backend

🎨 Streamlit Dashboard

🐳 Docker Deployment

📋 Project Overview

Customer reviews contain valuable insights for businesses. This project analyzes 23,486 real customer reviews to classify sentiment into:

✅ Positive

😐 Neutral

❌ Negative

Using NLP techniques and machine learning, the system can automatically determine the sentiment of new reviews in real time.

🎯 Key Features

✔ Interactive Dashboard
Explore dataset insights using Streamlit visualizations.

✔ Real-Time Sentiment Prediction
Enter any review and instantly get predicted sentiment.

✔ Batch Review Analysis
Upload CSV files and analyze multiple reviews at once.

✔ REST API Integration
Use FastAPI endpoints to integrate sentiment prediction into other applications.

✔ Model Performance Insights
View confusion matrix, evaluation metrics, and feature importance.

✔ Docker Deployment
Easily deploy using containerized environments.

🏗️ Project Structure
womens-clothing-sentiment
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── sentiment_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── utils.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_metrics.json
│
├── api/
│   └── app.py
│
├── frontend/
│   └── streamlit_app.py
│
├── requirements.txt
├── Dockerfile
└── README.md
🚀 Quick Start
1️⃣ Clone Repository
git clone https://github.com/wardaadan03-netizen/women-clothing-sentiment.git
cd womens-clothing-sentiment
2️⃣ Create Virtual Environment
python -m venv venv

Activate environment:

Windows

venv\Scripts\activate

Mac / Linux

source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Download Dataset

Dataset: Women's E-Commerce Clothing Reviews

Download from Kaggle and place it inside:

data/raw/
🧪 Run the Applications
▶ Streamlit Web App
streamlit run frontend/streamlit_app.py

Open in browser:

http://localhost:8501
⚡ FastAPI Backend
uvicorn api.app:app --reload

Access API:

http://localhost:8000

Interactive API documentation:

http://localhost:8000/docs
🐳 Docker Deployment

Build Docker image:

docker build -t clothing-sentiment .

Run Streamlit container:

docker run -p 8501:8501 clothing-sentiment streamlit

Run FastAPI container:

docker run -p 8000:8000 clothing-sentiment api
📊 Dataset

Women's E-Commerce Clothing Reviews

Source: Kaggle
Total Reviews: 23,486

Dataset Features
Feature	Description
Clothing ID	Product identifier
Age	Customer age
Review Title	Title of review
Review Text	Full review text
Rating	Rating from 1–5
Recommended IND	Whether customer recommends the product
Positive Feedback Count	Helpful votes
Division Name	Clothing division
Department Name	Department category
Class Name	Product category
🤖 Machine Learning Model

The model uses TF-IDF feature extraction combined with Logistic Regression.

Model Configuration

TF-IDF Features: 5000

N-grams: (1,2)

Regularization: L2

Algorithm: Logistic Regression

📈 Model Performance
Metric	Score
Accuracy	89.5%
Precision	0.92
Recall	0.96
F1 Score	0.94

Cross-Validation Score:

~89% (5-Fold CV)
🌐 API Endpoints
Health Check
GET /

Returns API status.

Predict Single Review
POST /predict

Example request:

{
"text": "I love this dress!"
}
Predict Multiple Reviews
POST /predict/batch

Example request:

{
"texts": ["Great product!", "Too small", "Average quality"]
}
Model Information
GET /model/info

Returns model configuration and metrics.

🎯 Use Cases

🛒 E-Commerce Platforms
Automatically analyze customer reviews.

📞 Customer Support
Prioritize negative feedback for faster responses.

📦 Product Insights
Understand product sentiment trends.

📊 Market Research
Identify customer preferences and issues.

🔧 Future Improvements

Deep Learning models (LSTM / BERT)

Multi-language sentiment analysis

Business analytics dashboard

Real-time streaming sentiment analysis

A/B testing framework

📝 License

This project is licensed under the MIT License.

👩‍💻 Author

Warda Adan

📧 Email: wardaadan03@gmail.com

🔗 GitHub:
https://github.com/wardaadan03-netizen/women-clothing-sentiment

🙏 Acknowledgments

Dataset by Nicolas Potato (Kaggle)

Inspired by open-source NLP and machine learning projects

⭐ If you found this project useful, consider giving it a star!
