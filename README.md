# Fiona Heerink Data Science Portfolio 

Welcome! This portfolio serves as an overview of my data science projects.
My work focuses on statistical methods, applied machine learning, and NLP, 
with a strong interest in Generative AI and Retrieval-Augmented Generation (RAG).

## Featured Projects

### 1. Anomaly Detection in Global Burden of Disease Data
**Techniques:** Exploratory Data Analysis, Isolation Forest, One-Class SVM, PCA  
**Tools:** Python, pandas, scikit-learn, Plotly  
**Data:** IHME Global Burden of Disease (GBD) 2025 release

This project applies statistical and machine learning–based anomaly detection
methods to country-level cause-of-death profiles from the Global Burden of Disease
study. The goal is to identify countries with unusual mortality patterns and
demonstrate robust, interpretable anomaly detection approaches.

🔗 **See this project here:** https://github.com/fionaheerink/gbd-anomaly-detection

![Preview of IQR outlier map](https://raw.githubusercontent.com/fionaheerink/gbd-anomaly-detection/main/docs/figures/map_IQR_preview.png)

---

## Other Projects

### 2. Customer Segmentation with Clustering  
**Techniques:** Feature engineering, K-Means, elbow method, silhouette score, hierarchical clustering + dendrogram, PCA, t-SNE  
**Tools:** Python, pandas, scikit-learn, Matplotlib, Seaborn, SciPy  
**Data:** E-commerce transactions dataset

This project builds an end-to-end customer segmentation pipeline using unsupervised learning. I engineered customer-level behavioural features and compared multiple clustering validation methods to identify meaningful customer groups.

🔗 **Link to project:** https://github.com/fionaheerink/customer-segmentation-clustering

<img src="https://raw.githubusercontent.com/fionaheerink/customer-segmentation-clustering/main/customer_clusters_tsne.png" width="600">

### 3. Predictive Modelling (Supervised Learning)
**Techniques:** EDA, preprocessing, one-hot encoding, train-test split, XGBoost, neural networks, hyperparameter tuning, model evaluation, feature importance  
**Tools:** Python, pandas, scikit-learn, XGBoost, TensorFlow/Keras  
**Data:** Proprietary business dataset (not publicly shareable)

This project compares supervised learning approaches for predicting an outcome variable using structured business data. Models were evaluated using accuracy, precision, recall, F1-score, confusion matrix, ROC curve, and AUC. Feature importance visualisations were used to support interpretation, and model performance was compared across different feature sets.

🔗 **Link to project:** https://github.com/fionaheerink/supervised-learning-predictive

### 4. NLP Topic Modelling of Customer Reviews
**Techniques:** text cleaning, tokenisation, stopword removal, word frequency analysis, word clouds, filtering negative reviews, BERTopic, clustering visualisations, emotion classification (BERT), Phi4-mini-instruct LLM-based topic extraction and recommendations, LDA (Gensim)  
**Tools:** Python, pandas, NLTK, BERTopic, Hugging Face Transformers, Gensim, Matplotlib/Seaborn, Plotly  
**Data:** Proprietary review data (not publicly shareable)

This project analyses customer reviews from two platforms to identify common themes and recurring issues, with a focus on negative feedback. Topic modelling and emotion analysis were used to explore patterns across locations and to compare results across multiple modelling approaches. An LLM was used with prompt-based topic extraction to produce recommendations aligned with the most frequent topics.

🔗 **Link to project:** https://github.com/fionaheerink/nlp-topic-modelling

### 5. Bank of England Employer Project – Earnings Call NLP + RAG (Group Project)
**Tools:** Python, Pandas, HuggingFace, LangChain, Chroma, Sentence-Transformers, XGBoost  
**Project type:** Industry-sponsored group project, completed as part of the Cambridge Career Accelerator in Data Science with AI and ML  
**Data:** Unstructured PDF-files from quarterly bank earnings calls, Q&A sessions.

In this project, I collaborated in a diverse, online team of seven to analyse quarterly bank earnings call transcripts and explored whether language signals can complement prudential indicators such as CET1, LCR, NSFR and LDR. We investigated language signals as potential early warning-indicators, such as sentiment, disagreement, and evasiveness.

My main responsibility was building the **Retrieval Augmented Generation (RAG)** component: chunking and embedding transcripts, storing them in a vector database, and creating a chatbot-style Q&A workflow with metadata-based retrieval from unstructured PDF data.

**Link to project:** https://github.com/fionaheerink/nlp-employer-project

---

## More Projects (upcoming)

- **Time-series forecasting**  
  Sales and demand forecasting using Nielsen BookScan data.
