import streamlit as st
import pandas as pd
import joblib
import inspect
import os

# --- 1. Custom Styling (Dark Mode with Teal/Amber Accents) ---
def apply_custom_css():
    st.markdown("""
        <style>
        /* General page configuration */
        .stApp {
            background-color: #1a1a1a; /* Dark background */
            color: #f0f0f0; /* Light text */
        }
        
        /* Custom header color (Teal) */
        h1 {
            color: #00bcd4; /* Cyan/Teal for Main Title */
        }
        h2 {
            color: #b2ebf2; /* Lighter Teal for Subheaders */
        }
        
        /* Style the Prediction Container/Columns */
        div[data-testid="stVerticalBlock"] {
            border-radius: 10px;
        }

        /* Customize Metric Boxes for Output */
        [data-testid="stMetric"] > div {
            background-color: #2e2e2e; /* Slightly lighter block */
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #333333;
        }

        /* FRAUD ALERT Styling (Amber/Gold) */
        .st-emotion-cache-1wv5f76 { /* Target the alert box specifically for error */
            background-color: #3d3d00; /* Dark Amber/Gold background for error state */
            border-left: 5px solid #ffc107; /* Bright Amber border */
            color: #ffc107;
            font-weight: bold;
        }
        
        /* SAFE Styling (Teal) */
        .st-emotion-cache-121vs35 { /* Target success alert */
            background-color: #003333; 
            border-left: 5px solid #00bcd4;
            color: #00bcd4;
            font-weight: bold;
        }
        
        /* Button Style */
        .stButton>button {
            width: 100%;
            background-color: #00bcd4;
            color: black;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

# --- 2. Model Loading and Setup ---

@st.cache_resource
def load_best_model():
    """Loads the best-performing XGBoost model."""
    try:
        # NOTE: Model file name has been fixed back to the expected name for general use.
        model = joblib.load("Ensemble_Fraud_Detection_Model.joblib")
        return model
    except Exception as e:
        # Do not halt execution, let the UI display the error
        # Do not halt execution, let the UI display the error
        st.session_state['model_error'] = f"Model Error: {e}. Check if 'Fraud_Detection_Model.joblib' exists."
        return None

MODEL = load_best_model()

# Define the exact order of features the model was trained on
FEATURE_ORDER = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
    'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud',
    'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'
]

# --- 3. Input Handling ---
# The inputs are now placed directly into the main page columns
def get_user_input(col):
    """Collects all 11 features using widgets in the specified column."""
    
    with col:
        st.subheader("Transaction Parameters (Input)")

        # Initialize dictionary to store inputs
        data = {}
        
        # 1. Transaction Type Selector (Used for OHE)
        transaction_type = st.selectbox(
            "1. Transaction Type (CRITICAL)",
            ('TRANSFER', 'CASH_OUT', 'PAYMENT', 'DEBIT'),
            index=0, # Default to TRANSFER for fraud testing
            help="Fraud is heavily concentrated in TRANSFER/CASH_OUT."
        )

        # 2. Core Features
        step = st.slider("2. Time Step (Hour)", 1, 743, 10, help="Maps to the hour of the week (1-743).")
        amount = st.number_input("3. Amount", min_value=0.01, value=50000.00, step=1000.0, format="%.2f")
        
        # 3. Balances
        st.markdown("---")
        st.markdown("**Originator Balances**")
        oldbalanceOrg = st.number_input("4. Old Balance (Origin)", min_value=0.0, value=50000.00, step=100.0, format="%.2f")
        newbalanceOrig = st.number_input("5. New Balance (Origin)", min_value=0.0, value=0.00, step=100.0, format="%.2f")
        
        st.markdown("**Destination Balances**")
        oldbalanceDest = st.number_input("6. Old Balance (Dest)", min_value=0.0, value=0.00, step=100.0, format="%.2f")
        newbalanceDest = st.number_input("7. New Balance (Dest)", min_value=0.0, value=50000.00, step=1000.0, format="%.2f")
        
        # 4. Flagged Status
        st.markdown("---")
        isFlaggedFraud = st.selectbox(
            "8. isFlaggedFraud", 
            (0, 1), 
            index=0, 
            help="Whether the transaction was flagged by the system (1) or not (0)."
        )

    # --- Combine and Finalize Data ---
    data = {
        'step': step,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'isFlaggedFraud': isFlaggedFraud,
        # One-Hot Encoding based on selection
        'CASH_OUT': 1.0 if transaction_type == 'CASH_OUT' else 0.0,
        'DEBIT': 1.0 if transaction_type == 'DEBIT' else 0.0,
        'PAYMENT': 1.0 if transaction_type == 'PAYMENT' else 0.0,
        'TRANSFER': 1.0 if transaction_type == 'TRANSFER' else 0.0
    }
    
    # Convert to DataFrame, ensuring the order matches the trained model
    input_df = pd.DataFrame([data], columns=FEATURE_ORDER)
    return input_df


# --- 4. Prediction Function ---

def make_prediction(df):
    """Uses the loaded model to make a prediction."""
    if MODEL is None:
        return None, None
    try:
        prob_fraud = MODEL.predict_proba(df)[:, 1][0]
        # Use the standard classification threshold of 0.5
        prediction = 1 if prob_fraud > 0.5 else 0
        return prediction, prob_fraud
    except Exception as e:
        st.error(f"Prediction failed due to model input error: {e}")
        return None, None


# --- 5. Main Execution and UI ---

apply_custom_css()
st.set_page_config(layout="wide", page_title="Fraud Model Tester")

st.title("💸 AI Online Payment Fraud Detection System")
st.markdown("---")

if 'model_error' in st.session_state:
    st.error(st.session_state['model_error'])
    st.stop()
    
# Create two main columns: one for input, one for output/results
col_input, col_output = st.columns(2)

# --- Collect Input ---
input_data = get_user_input(col_input)

# --- Place Button in Input Column ---
with col_input:
    st.markdown("---")
    if st.button("Run Fraud Analysis", key="run_analysis_btn"):
        st.session_state['run_prediction'] = True
    
# --- Prediction and Output Logic ---
if st.session_state.get('run_prediction', False):
    
    prediction, prob_fraud = make_prediction(input_data)
    
    with col_output:
        st.subheader("Analysis Result (Output)")
        
        if prediction is not None:
            # Display prediction result in a stylized box
            if prediction == 1:
                st.error("🚨 HIGH-RISK FRAUD ALERT", icon="🚫")
            else:
                st.success("✅ LOW-RISK: TRANSACTION IS SAFE", icon="🔒")

            # Display metrics
            col_metric_1, col_metric_2 = st.columns(2)
            
            with col_metric_1:
                st.metric(label="Predicted Status", value="FRAUD" if prediction == 1 else "SAFE", delta_color="inverse")
            
            with col_metric_2:
                # Color code the confidence score: redder for high fraud probability
                st.metric(label="Confidence Score", value=f"{prob_fraud*100:.2f} %", 
                         delta=f"{prob_fraud*100:.2f} %", delta_color="normal" if prediction == 1 else "inverse")
                
            st.markdown("---")
            
            # Display Data Frame in the output column
            st.markdown("**Transaction Vector Sent to Model:**")
            st.dataframe(input_data.T, use_container_width=True)

# Reset state flag after showing results
st.session_state['run_prediction'] = False
