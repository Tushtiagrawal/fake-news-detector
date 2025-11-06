
# import streamlit as st
# import joblib
# import pandas as pd # Import pandas (good practice, though not strictly required here)

# # --- 1. Fix Path Syntax and Load Assets ---

# # Note: The 'r' before the string creates a raw string, preventing the Unicode escape error.
# # We load the vectorizer
# try:
#     vectorization = joblib.load(r"C:\Users\tusht\Downloads\verctorization.jb")
# except FileNotFoundError:
#     st.error("Vectorizer file not found. Check the path: verctorization.jb")
#     st.stop()
# except Exception as e:
#     st.error(f"Error loading vectorizer: {e}")
#     st.stop()


# # Load all four models into a dictionary, fixing the path syntax for all
# models = {}
# try:
#     models["LR"] = joblib.load(r"C:\Users\tusht\Downloads\LR_model.jb")
#     models["DT"] = joblib.load(r"C:\Users\tusht\Downloads\DT_model.jb")
#     models["GBC"] = joblib.load(r"C:\Users\tusht\Downloads\GBC_model.jb")
#     models["RFC"] = joblib.load(r"C:\Users\tusht\Downloads\RFC_model.jb")
# except Exception as e:
#     st.error(f"Error loading one or more model files: {e}")
#     st.stop()

# # --- 2. Streamlit Application UI and Logic ---

# st.title("📰 Fake News Detector")
# st.markdown("Enter a news article or snippet below to check whether it is likely **Fake** or **Real**.")

# # Model Selector (Allows the user to choose which model to use)
# model_choice = st.selectbox(
#     "Select the Model for Prediction:",
#     list(models.keys()),
#     index=0,
#     help="Choose between Logistic Regression (LR), Decision Tree (DT), Gradient Boosting (GBC), or Random Forest (RFC)."
# )
# selected_model = models[model_choice]

# # Text Area for User Input
# news_input = st.text_area("News Article:", height=200, placeholder="Paste your article here...")


# # Prediction Button
# if st.button(f"Check with {model_choice}"):
#     if news_input and news_input.strip():
#         try:
#             # 1. Transform the input text using the loaded vectorizer
#             # The input must be passed as a list/array of strings: [news_input]
#             transform_input = vectorization.transform([news_input])

#             # 2. Predict using the selected model
#             prediction = selected_model.predict(transform_input)

#             # 3. Display the result
#             if prediction[0] == 1:
#                 st.success("✅ The news is classified as **REAL**!")
#             else:
#                 st.error("🛑 The news is classified as **FAKE**!")

#             st.balloons()

#         except Exception as e:
#             st.exception(f"An unexpected error occurred during prediction: {e}")

#     else:
#         st.warning("⚠️ Please enter some text into the box to analyze.")

# # Add some helpful context
# st.markdown(
#     """
#     ---
#     *Disclaimer: This detector uses pre-trained machine learning models and may not be 100% accurate. 
#     Always cross-reference information from multiple reliable sources.*
#     """
# )


import streamlit as st
import joblib
import os

# --- Load Vectorizer ---
try:
    vectorization = joblib.load("verctorization.jb")
except FileNotFoundError:
    st.error("Vectorizer file not found. Ensure it's in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading vectorizer: {e}")
    st.stop()

# --- Load Models ---
models = {}
try:
    models["LR"] = joblib.load("LR_model.jb")
    models["DT"] = joblib.load("DT_model.jb")
    models["GBC"] = joblib.load("GBC_model.jb")
    models["RFC"] = joblib.load("RFC_model.jb")
except Exception as e:
    st.error(f"Error loading one or more model files: {e}")
    st.stop()

# --- Streamlit UI ---
st.title("📰 Fake News Detector")
st.markdown("Enter a news article or snippet below to check whether it is likely **Fake** or **Real**.")

model_choice = st.selectbox(
    "Select the Model for Prediction:",
    list(models.keys()),
    index=0,
    help="Choose between Logistic Regression (LR), Decision Tree (DT), Gradient Boosting (GBC), or Random Forest (RFC)."
)
selected_model = models[model_choice]

news_input = st.text_area("News Article:", height=200, placeholder="Paste your article here...")

if st.button(f"Check with {model_choice}"):
    if news_input.strip():
        try:
            transform_input = vectorization.transform([news_input])
            prediction = selected_model.predict(transform_input)
            if prediction[0] == 1:
                st.success("✅ The news is classified as **REAL**!")
            else:
                st.error("🛑 The news is classified as **FAKE**!")
            st.balloons()
        except Exception as e:
            st.exception(f"An unexpected error occurred: {e}")
    else:
        st.warning("⚠️ Please enter some text into the box to analyze.")

st.markdown("---")
st.caption("*Disclaimer: This detector uses pre-trained machine learning models and may not be 100% accurate.*")


