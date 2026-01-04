"""
AutoJudge: Problem Difficulty Predictor
A simple web interface to predict programming problem difficulty.
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix

# --- Page Configuration ---
st.set_page_config(
    page_title="AutoJudge - Problem Difficulty Predictor",
    page_icon=None,
    layout="centered"
)

# --- Load Models ---
@st.cache_resource
def load_models():
    """Load all required models from the models directory."""
    classifier = joblib.load("models/classifier.pkl")
    regressor = joblib.load("models/reg_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return classifier, regressor, tfidf, label_encoder

# --- Text Preprocessing ---
def clean_text(text):
    """Clean text while keeping important features."""
    if not isinstance(text, str) or text == '':
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_heuristic_features(combined_text, desc_clean, input_clean, output_clean):
    """Extract numerical features from text."""
    features = {}
    
    # Length-based features
    features['text_length'] = len(combined_text)
    words = combined_text.split()
    features['word_count'] = len(words)
    features['sentence_count'] = combined_text.count('.') + combined_text.count('!') + combined_text.count('?') + 1
    features['avg_word_length'] = features['text_length'] / (features['word_count'] + 1)
    
    # Description-specific lengths
    features['desc_length'] = len(desc_clean)
    features['input_desc_length'] = len(input_clean)
    features['output_desc_length'] = len(output_clean)
    
    # Math symbols count
    math_pattern = r'[\$\^\{\}\\\[\]≤≥∑∏∫]|\\leq|\\geq|\\sum|\\prod'
    features['math_symbol_count'] = len(re.findall(math_pattern, combined_text))
    
    # Constraint patterns
    constraint_pattern = r'\d+\s*[≤<>≥]\s*\w+\s*[≤<>≥]\s*\d+'
    features['constraint_count'] = len(re.findall(constraint_pattern, combined_text))
    
    # Algorithmic keyword detection
    features['has_graph'] = 1 if re.search(r'\b(graph|node|edge|vertex|tree|path|cycle)\b', combined_text) else 0
    features['has_dp'] = 1 if re.search(r'\b(dynamic|optimal|maximize|minimize|subproblem)\b', combined_text) else 0
    features['has_recursion'] = 1 if re.search(r'\b(recursive|recursion|recurrence)\b', combined_text) else 0
    features['has_sort'] = 1 if re.search(r'\b(sort|sorted|sorting|order)\b', combined_text) else 0
    features['has_binary'] = 1 if re.search(r'\b(binary|search|mid|bisect)\b', combined_text) else 0
    features['has_matrix'] = 1 if re.search(r'\b(matrix|grid|2d|row|column)\b', combined_text) else 0
    
    return features

def predict_difficulty(description, input_desc, output_desc, classifier, regressor, tfidf, label_encoder):
    """Predict difficulty class and score for a problem."""
    
    # Clean the text
    desc_clean = clean_text(description)
    input_clean = clean_text(input_desc)
    output_clean = clean_text(output_desc)
    
    # Create combined text (simulating the notebook's format)
    combined_text = f"{desc_clean} {input_clean} {output_clean}"
    combined_text = re.sub(r'\s+', ' ', combined_text).strip()
    
    # Extract TF-IDF features
    tfidf_features = tfidf.transform([combined_text])
    
    # Extract heuristic features
    heuristic = extract_heuristic_features(combined_text, desc_clean, input_clean, output_clean)
    heuristic_df = pd.DataFrame([heuristic])
    heuristic_sparse = csr_matrix(heuristic_df.values)
    
    # Combine features
    X = hstack([tfidf_features, heuristic_sparse])
    
    # Predict class and raw score
    class_pred = classifier.predict(X)[0]
    score_pred = regressor.predict(X)[0]
    
    # Decode class label
    class_name = label_encoder.inverse_transform([class_pred])[0].upper()
    
    # Adjust score based on predicted class to ensure consistency
    # Score ranges: Easy (1-3.5), Medium (3.5-6.5), Hard (6.5-10)
    if class_name == "HARD":
        # Scale score to 6.5-10 range
        score_adjusted = 6.5 + (score_pred - 1) * (3.5 / 9)
        score_adjusted = max(6.5, min(10.0, score_adjusted))
    elif class_name == "MEDIUM":
        # Scale score to 3.5-6.5 range
        score_adjusted = 3.5 + (score_pred - 1) * (3.0 / 9)
        score_adjusted = max(3.5, min(6.5, score_adjusted))
    else:  # EASY
        # Scale score to 1-3.5 range
        score_adjusted = 1.0 + (score_pred - 1) * (2.5 / 9)
        score_adjusted = max(1.0, min(3.5, score_adjusted))
    
    return class_name, round(score_adjusted, 1)

# --- Main Application ---
def main():
    st.title("AutoJudge")
    st.subheader("Programming Problem Difficulty Predictor")
    
    st.write("Enter a problem description and input/output specification to predict its difficulty level.")
    
    st.markdown("---")
    
    # Load models
    try:
        classifier, regressor, tfidf, label_encoder = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Make sure all model files exist in the 'models/' directory.")
        return
    
    # Input fields
    description = st.text_area(
        "Problem Description",
        height=200,
        placeholder="Enter the problem statement here..."
    )
    
    input_desc = st.text_area(
        "Input Description",
        height=120,
        placeholder="Enter input format description here..."
    )
    
    output_desc = st.text_area(
        "Output Description",
        height=120,
        placeholder="Enter output format description here..."
    )
    
    # Predict button
    if st.button("Predict Difficulty", type="primary"):
        if not description.strip():
            st.warning("Please enter a problem description.")
        else:
            with st.spinner("Analyzing problem..."):
                try:
                    difficulty_class, difficulty_score = predict_difficulty(
                        description, input_desc, output_desc,
                        classifier, regressor, tfidf, label_encoder
                    )
                    
                    st.markdown("---")
                    st.subheader("Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(label="Difficulty Class", value=difficulty_class)
                    
                    with col2:
                        st.metric(label="Difficulty Score", value=f"{difficulty_score:.1f}/10")
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()
