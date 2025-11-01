# 🎬 Movie Success Prediction using Machine Learning

## 📌 Project Overview

This project predicts movie success using machine learning by analyzing metadata and text features from the TMDB Movie Dataset. We define success as movies that either achieve high box office revenue (above median) or receive critical acclaim (rating > 6.5/10).

**Key Achievement:** Random Forest model achieved **86.8% F1-Score** by combining structured data with natural language processing.

---

## 🎯 Objectives

- Predict movie success using historical data
- Identify key factors that contribute to a movie's success
- Compare multiple ML algorithms and select the best performer
- Extract insights from movie descriptions using NLP

---

## 📊 Datasets

We use **2 datasets** for comprehensive analysis:

### 1. TMDB Movie Metadata (Primary Dataset)
**Source:** [TMDB on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

- **Total Movies:** 4,803
- **Features:** 20 columns (budget, revenue, popularity, vote_average, vote_count, runtime, overview, etc.)
- **Purpose:** Model training, feature engineering, and predictions
- **Target Variable:** Binary (Success/Not Success)
- **Success Rate:** 65.2%

### 2. Wikipedia Movie Plots (NLP Dataset)
**Source:** [Wikipedia Movie Plots on Kaggle](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)

- **Total Movies:** 34,000+
- **Features:** 7 columns (Release Year, Title, Origin/Ethnicity, Director, Cast, Genre, Plot)
- **Purpose:** NLP analysis, text preprocessing, sentiment analysis, word cloud generation
- **Usage:** Enhanced text feature engineering and plot-based insights

### Features Used in Final Model:
- **Numerical (4):** Budget, Popularity, Vote Count, Runtime
- **Text (100):** TF-IDF features from movie overviews
- **NLP Engineered (10):** Sentiment polarity, subjectivity, text complexity metrics
- **Total:** 114 features

---

## 🔬 Methodology

### 1. **Exploratory Data Analysis (EDA)**
- Dataset overview and missing value analysis
- Revenue, budget, and rating distributions
- Correlation analysis between features
- Visualization of success patterns

### 2. **Natural Language Processing (NLP)**
- **Wikipedia dataset analysis:** Explored 34,000+ movie plots
- Text preprocessing and cleaning
- Word frequency and n-gram analysis (bigrams, trigrams)
- Word cloud visualizations (successful vs. unsuccessful movies)
- TF-IDF vectorization (top 100 features) on TMDB overviews
- Sentiment analysis using TextBlob (polarity and subjectivity)
- Text complexity metrics (word length, sentence structure, lexical diversity)
- Feature engineering: 10 additional NLP features applied to TMDB data

### 3. **Data Preprocessing**
- Missing value imputation (median for numerical, empty string for text)
- Feature scaling using StandardScaler
- Train-test split (80-20) with stratification

### 4. **Model Training & Evaluation**

We trained and compared 4 different models:

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| **Random Forest** 🏆 | **82.9%** | **86.8%** | Fast |
| Logistic Regression | 81.1% | 85.3% | Very Fast |
| Neural Network | 77.9% | 83.2% | Slow |
| SVM | 76.4% | 82.4% | Medium |

---

## 🏆 Results

### Best Model: Random Forest Classifier

**Performance Metrics:**
- **Accuracy:** 82.9%
- **F1-Score:** 86.8%
- **Precision:** High
- **Recall:** High

### Top 5 Most Important Features:
1. **Vote Count** (26.5%) - Number of user votes
2. **Popularity** (25.7%) - Popularity score
3. **Budget** (16.8%) - Production budget
4. **Runtime** (7.8%) - Movie duration
5. Text features from movie descriptions

---

## 💡 Key Insights

✅ **Budget and popularity are strong success predictors** - Movies with higher budgets and pre-release popularity tend to succeed

✅ **Text features matter** - Movie descriptions contain valuable signals about success

✅ **Ensemble methods outperform** - Random Forest beat linear models and neural networks

✅ **Critical acclaim vs. Box Office** - Success can be achieved through either high revenue or high ratings

✅ **Vote count is the strongest predictor** - Audience engagement is crucial

---

## 🛠️ Technologies Used

### Languages & Libraries
- **Python 3.11**
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
- **Deep Learning:** TensorFlow/Keras
- **NLP:** TextBlob, TF-IDF Vectorizer, WordCloud

### Environment
- **Platform:** Kaggle Notebooks (2x Tesla T4 GPUs)
- **IDE:** VS Code with Jupyter extension

---

## 📁 Project Structure

```
ML-Project/
├── movie_success_prediction.ipynb    # Main Jupyter notebook
├── README.md                          # Project documentation
├── data/
│   ├── tmdb_5000_movies.csv          # TMDB dataset
│   └── wiki_movie_plots_deduped.csv  # Wikipedia plots (for NLP)
└── outputs/                           # (Generated during execution)
    ├── visualizations/                # EDA and model comparison plots
    └── results/                       # Model performance metrics
```

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/laxmikhilnani/ML-Project.git
cd ML-Project
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow textblob wordcloud
```

### 3. Download Datasets (Both Required)
Download both datasets from Kaggle:

**Dataset 1: TMDB Movie Metadata**
- URL: [https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- File: `tmdb_5000_movies.csv`

**Dataset 2: Wikipedia Movie Plots**
- URL: [https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
- File: `wiki_movie_plots_deduped.csv`

Place both CSV files in the project root or update paths in the notebook.

### 4. Run the Notebook
```bash
jupyter notebook movie_success_prediction.ipynb
```

Or open in VS Code with Jupyter extension and run all cells sequentially.

---

## 📈 Visualizations

The project includes comprehensive visualizations:

- **EDA Charts:** Revenue distributions, budget analysis, correlation heatmaps
- **NLP Visualizations:** Word clouds, n-gram analysis, text length distributions
- **Model Comparison:** Bar charts, confusion matrices, polar plots
- **Feature Importance:** Horizontal bar charts showing top predictors

---

## 🔮 Future Enhancements

### Model Improvements
- [ ] Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- [ ] Try XGBoost and LightGBM
- [ ] Implement ensemble stacking
- [ ] Add cross-validation for robust evaluation

### Feature Engineering
- [ ] Include cast and crew information
- [ ] Add genre-based features
- [ ] Incorporate release timing (season, holidays)
- [ ] Social media sentiment analysis

### Additional Analysis
- [ ] Regional performance predictions
- [ ] Genre-specific models
- [ ] ROI (Return on Investment) prediction
- [ ] Time series analysis of movie trends

---

## 📝 Key Takeaways

1. **Data-driven decisions work** - ML can predict movie success with 86.8% F1-score
2. **Text + Metadata is powerful** - Combining structured and unstructured data improves predictions
3. **Ensemble methods excel** - Random Forest outperformed individual models
4. **Feature engineering matters** - Creating meaningful features from text data boosted performance
5. **Real-world applications** - Studios can use this for investment and marketing decisions

---

## 👨‍💻 Author

**Laxmi Khilnani**
- GitHub: [@laxmikhilnani20](https://github.com/laxmikhilnani)
- Project: ML Movie Success Prediction

---

## 📄 License

This project is open-source and available for educational purposes.

---

## 🙏 Acknowledgments

- **TMDB** for providing the comprehensive movie metadata dataset
- **Wikipedia** for the extensive movie plots dataset
- **Kaggle** for hosting both datasets and providing GPU resources (2x Tesla T4)
- **Scikit-learn** community for excellent ML libraries
- **TensorFlow** team for deep learning framework
- **TextBlob** for NLP and sentiment analysis tools

---

## 📧 Contact

For questions, suggestions, or collaborations:
- Open an issue on GitHub
- Connect via GitHub profile

---

**⭐ If you found this project helpful, please give it a star!**

---

*Last Updated: November 2025*
# ML_project
