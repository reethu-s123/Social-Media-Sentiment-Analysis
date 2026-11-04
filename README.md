# ⚡ AI Sentiment Pro: Large-Scale Social Media Analytics Platform

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-Distributed-orange.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)
![NLP](https://img.shields.io/badge/NLP-VADER-green.svg)

## 🚀 Project Overview
This project is a high-performance, distributed Sentiment Analysis system designed to handle large-scale social media data (300,000+ records). Built for the **MCA 3rd Semester** curriculum, it focuses on predicting public opinion trends for **Movie and Product Reviews** on platforms like Twitter.

By leveraging **Apache Spark's** distributed architecture and **NLP (Natural Language Processing)**, the system bypasses the limitations of traditional single-machine processing, enabling real-time insights into global sentiment spikes.

## 🎯 Key Objectives
- **Distributed Processing:** Simulate a multi-node cluster environment using Apache Spark for Big Data handling.
- **Accurate Classification:** Use the VADER Lexicon to categorize text into Positive, Negative, and Neutral sentiments.
- **Scalability:** Handle massive CSV/JSON datasets (HDFS ready) with optimized character encoding.
- **Visual Intelligence:** Provide an interactive dashboard for stakeholders to track "Public Mood" trends.

## 🛠️ Tech Stack
- **Core Engine:** Apache Spark (PySpark)
- **Natural Language Processing:** NLTK, VADER Lexicon
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Plotly Express, Plotly Graph Objects
- **Web Interface:** Streamlit (Custom CSS Glassmorphism UI)
- **Data Source:** Twitter API / Kaggle Social Media Datasets

## 📂 Project Structure
```text
├── app.py              # Main Streamlit Application (Login + Dashboard)
├── requirements.txt    # List of Python dependencies
├── README.md           # Project Documentation
└── raw_tweets.csv      # Sample Big Data file (3 Lakh+ records)

⚙️ How It Works
Secure Login: Access control via a professional login interface.
Live Engine: Users can input specific reviews to see real-time polarity scores and gauge visualizations.
Batch Processing: The system ingests large CSV files, cleans the text using Regex (removing URLs, handles, emojis), and processes them in parallel chunks.
Analytics Dashboard: Generates Time-Series trends, Sentiment Distribution (Pie Charts), and Polarity Histograms.

Developed with ❤️ for Big Data Research.
