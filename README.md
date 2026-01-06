AutoJudge

 App link: https://autojudge-difficulty-predictor.streamlit.app/

This project uses machine learning to estimate how difficult a programming problem is. Given a problem statement, it outputs a difficulty category (easy, medium, or hard) along with a numerical score from 1 to 10.


Dataset

The training data comes from a custom collection of competitive programming problems.
Total entries: 4112
Each entry contains:
- Title
- Description
- input_description
- output_description
- sample_io
- problem_class
- problem_score
- url


Text Preprocessing

All text fields (title, description, input/output specs) were merged into one combined text. Missing or empty values were handled by substituting empty strings.


Feature Extraction

Keyword frequency was computed for common algorithmic terms such as greedy, dp, dfs, bfs, graph, tree, matrix, etc.
TF-IDF was applied to convert textual content into numerical vectors (limited to top 5000 terms).
Additional numeric features were derived including text length, constraint patterns, and presence of mathematical notation.
SMOTE was used to oversample minority classes during training to address class imbalance.


Classification

Data was partitioned into training (70%), validation (15%), and test (15%) sets.

Logistic Regression
Achieved approximately 48% accuracy.

Random Forest Classifier
Achieved 52.19% accuracy
Class-wise breakdown:
  Easy: 33.91%
  Hard: 82.47%
  Medium: 20.38%
F1-Score (Macro): 0.4860

Confusion Matrix (Random Forest Classification)
```
               Predicted
             easy  hard  medium
Actual easy    39    52     24
Actual hard    20   240     31
Actual medium  22   146     43
```

XGBoost Classifier
Achieved 50.729% accuracy
Class-wise breakdown:
  Easy: 33.91%
  Hard: 82.47%
  Medium: 20.38%
F1-Score : 0.41 for easy, 0.63 for hard and 0.32 for medium.
Confusion Matrix (xgboost Classification)

```
           Predicted
           easy  hard  medium
Actual     
easy        55    63     35
hard        15   291     83
medium      24   190     67
```



 Although running this xgboost classifier with GPU acceleration(on google colab) improved the accuracy to 54.729%.



Among the classifiers tested, Random Forest delivered the best overall performance. It showed strong results for Hard problems but struggled to correctly identify Medium difficulty problems.


Regression

Same data split was used (70% train, 15% validation, 15% test).

Linear Regression
RMSE score: approximately 2.8

XGBoost Regressor
MAE: 1.68
RMSE: 2.05
R²: 0.12

The high RMSE from Linear Regression suggests the mapping from text to difficulty score is inherently non-linear.
XGBoost handles this better by learning complex patterns in the data.

Dual-Head Model

I have also tried dual-head prediction architecture where a shared backbone learns common representations and a classification head predicts the problem difficulty class (Easy, Medium, Hard). The classification head achieved an overall accuracy of 51.64%, which is reasonable given the subjective and overlapping nature of difficulty labels.

Confusion Matrix (Dual-Head Classification)
```
           Predicted
           easy  hard  medium
Actual
easy        51    62     40
hard         9   311     69
medium      33   185     63
```
 
 The confusion matrix shows that the model predicts the Hard class most reliably, while Medium problems are often confused with Hard, indicating overlap in intermediate difficulty levels.

 The regression head achieved a MAE of 1.7 and an RMSE of 2.06.



Final Model Selection

After comparing all approaches, Random Forest Classifier was selected for categorization and XGBoost Regressor was chosen for score prediction as they yielded the best metrics.


Web Application

Built using Streamlit which handles both frontend and backend.

How it works:
- User enters problem description, input format, and output format
- Text is cleaned and transformed using saved TF-IDF vectorizer
- Heuristic features are extracted
- Models predict the class and score
- Results are displayed on screen

Libraries used:
- Streamlit for web interface
- Scikit-learn for classification
- XGBoost for regression
- joblib for model persistence
- numpy and scipy for numerical operations

Web Interface and Backend Integration

The web application is built entirely in Python using Streamlit, which provides both the user interface and backend functionality in a single framework.
When a user submits a problem description, the system cleans the text, applies TF-IDF vectorization using a pre-saved vectorizer, computes heuristic features, and passes everything to the trained models. The predicted difficulty class and score are then rendered on the page. Key libraries include Streamlit for the web layer, Scikit-learn and XGBoost for predictions, joblib for model loading, and numpy/scipy for numerical computations.
 

Running Locally

1. Clone this repository and navigate to the project folder

2. Install required packages:
   pip install -r requirements.txt

3. Launch the application:
   streamlit run app.py
   
   Open the URL shown in terminal in the browser (eg., http://localhost:8501)
 
 Video link: 

Submitted by:
Manasvi Sawaria,
Chemical Engineering, IIT Roorkee (2nd Year)
manasvi_s@ch.iitr.ac.in
