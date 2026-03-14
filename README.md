# 👕 Women's Clothing Reviews Sentiment Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

A complete end-to-end machine learning project for sentiment analysis on women's e-commerce clothing reviews. This project includes data exploration, model training, web application, and API deployment.

## 📋 Project Overview

This project analyzes customer reviews to determine sentiment (Positive, Neutral, Negative) using Natural Language Processing (NLP) techniques. The dataset contains 23,486 real customer reviews from a women's clothing e-commerce store.

### 🎯 Key Features

- **Interactive Dashboard**: Explore data with visualizations and statistics
- **Real-time Sentiment Analysis**: Enter any review and get instant sentiment prediction
- **Batch Processing**: Upload CSV files with multiple reviews for analysis
- **REST API**: Integrate sentiment analysis into other applications
- **Model Performance**: View confusion matrix, feature importance, and metrics
- **Docker Support**: Easy deployment with containerization

## 🏗️ Project Structure
womens-clothing-sentiment/
│
├── data/ # Dataset files
│ ├── raw/ # Original dataset
│ └── processed/ # Cleaned data
│
├── notebooks/ # Jupyter notebooks
│ └── 01_sentiment_analysis_complete.ipynb
│
├── src/ # Source code
│ ├── preprocessing.py # Text preprocessing
│ ├── model_training.py # Model training logic
│ ├── evaluation.py # Evaluation metrics
│ └── utils.py # Utility functions
│
├── models/ # Trained models
│ ├── logistic_regression.pkl
│ ├── tfidf_vectorizer.pkl
│ └── model_metrics.json
│
├── api/ # FastAPI application
│ └── app.py
│
├── frontend/ # Streamlit application
│ └── streamlit_app.py
│
├── requirements.txt # Dependencies
├── Dockerfile # Docker configuration
└── README.md # Project documentation

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Docker

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/womens-clothing-sentiment.git
cd womens-clothing-sentiment
Create virtual environment

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Download the dataset

Download from Kaggle

Place the CSV file in data/raw/

Run the Jupyter notebook

bash
jupyter notebook notebooks/01_sentiment_analysis_complete.ipynb
Run the Applications
Streamlit Web App
bash
streamlit run frontend/streamlit_app.py
Access at: http://localhost:8501

FastAPI Backend
bash
uvicorn api.app:app --reload
Access at: http://localhost:8000
API Docs: http://localhost:8000/docs

Docker Deployment
bash
# Build the image
docker build -t clothing-sentiment .

# Run Streamlit app
docker run -p 8501:8501 clothing-sentiment streamlit

# Run FastAPI app
docker run -p 8000:8000 clothing-sentiment api
Or use docker-compose:

bash
docker-compose up
📊 Dataset
Women's E-Commerce Clothing Reviews

Source: Kaggle

Size: 23,486 reviews

Features:

Clothing ID

Age

Review Title

Review Text

Rating (1-5)

Recommended IND

Positive Feedback Count

Division Name

Department Name

Class Name

🤖 Model
The project uses Logistic Regression with TF-IDF features:

TF-IDF Vectorizer: 5000 features, n-grams (1,2)

Model: Logistic Regression with L2 regularization

Accuracy: ~90% on test set

Cross-validation: 5-fold CV score: ~89%

📈 Performance
Metric	Score
Accuracy	89.5%
Precision (Positive)	0.92
Recall (Positive)	0.96
F1-Score (Positive)	0.94
🌐 API Endpoints
GET /
Health check endpoint

POST /predict
Predict sentiment for a single review

json
{
    "text": "I love this dress!"
}
POST /predict/batch
Predict sentiment for multiple reviews

json
{
    "texts": ["Great product!", "Too small", "Average quality"]
}
GET /model/info
Get model information

🎯 Use Cases
E-commerce Platforms: Automatically categorize customer feedback

Customer Service: Prioritize negative reviews for quick response

Product Analysis: Identify sentiment trends across products

Market Research: Understand customer preferences and pain points

🔧 Future Improvements
Implement deep learning models (LSTM, BERT)

Add multi-language support

Create dashboard for business analytics

Add real-time streaming analysis

Implement A/B testing framework

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

👥 Contributing

This project is primarily maintained by Warda Adan.

If you’d like to contribute in the future, you are welcome to:

Fork the repository

Make improvements or fixes

Submit a Pull Request

Currently, no external contributors are required, but contributions are appreciated.

Open a Pull Request

📧 Contact
Your Name - @email:wardaadan03@gmail.com

Project Link: https://github.com/wardaadan03-netizen/women-clothing-sentiment

🙏 Acknowledgments
Dataset by Nicolas Potato on Kaggle

Inspired by various NLP tutorials and projects

Built with ❤️ for the data science community