import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import pickle

def set_sidebar_style():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: rgba(0, 0, 0, 0);  /* Transparent background */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
def main():
    
    data = get_data()
    st.set_page_config(
        page_title="Heart Disease Prognosis",
        layout = "wide",
        initial_sidebar_state="expanded"
    )

    set_sidebar_style()

    page_background = """
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1710162734106-6932b5799f99?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-position: center;
    }

    [data-testid="stHeader"] {
    background: rgba(0,0,0,0);
    }

    [data-testid="stToolbar"] {
    right: 1rem;
    }
    </style>
    """

    st.markdown(page_background, unsafe_allow_html=True)

    with open("style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    st.title("Heart Disease Prognostic Model")

    input_data = sidebar(data)
    
    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    
    with col2:
        add_predictions(input_data)


def get_data():
    data = pd.read_csv(r"C:\Users\calli\OneDrive\Desktop\Strands\Deep Learning\Heart Disease Classification\Data\heart.csv")
    return data

def sidebar(data):
    st.sidebar.header("Values")

    ColorMinMax = st.markdown('''
        <style>
            div.stSlider > div[data-baseweb="slider"] > div[data-testid="stTickBar"] > div {
                background: rgb(1 1 1 / 0%);  /* Transparent background for the tick bar */
            }
        </style>''', unsafe_allow_html=True)

    Slider_Cursor = st.markdown('''
        <style>
            div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {
                background-color: rgb(14, 38, 74); /* Slider cursor color */
                box-shadow: rgb(14 38, 74 / 20%) 0px 0px 0px 0.2rem; /* Shadow for the slider cursor */
            }
        </style>''', unsafe_allow_html=True)

    Slider_Number = st.markdown('''
        <style>
            div.stSlider > div[data-baseweb="slider"] > div > div > div > div {
                color: rgb(14, 38, 74); /* Number color */
            }
        </style>''', unsafe_allow_html=True)

    slider_labels = [
    ("Age", "age"),
    ("Sex", "sex"),
    ("Chest Pain Type [0-3]", "cp"),
    ("Resting Blood Pressure (mm/Hg)", "trestbps"),
    ("Cholesterol (mg/dl)", "chol"),
    ("Fasting Blood Sugar > 120 mg/dl", "fbs"),
    ("Resting ECG Results [0-2]", "restecg"),
    ("Max Heart Rate", "thalach"),
    ("Exercise-Induced Angina", "exang"),
    ("Oldpeak (depression induced by exercise)", "oldpeak"),
    ("Slope of the Peak Exercise ST Segment [0-2]", "slope"),
    ("Number of Major Vessels [0-3]", "ca"),
    ("Thalassemia [1-3]", "thal"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        # Define min and max values for int-based sliders
        if key == "sex":
            # Use a select slider instead for sex to show Male/Female
            input_dict[key] = st.sidebar.select_slider(
                label,
                options=["Female", "Male"],  # Display labels
                value="Female"  # Default value as Female
            )
            # Map the selected label back to numerical values
            input_dict[key] = 1 if input_dict[key] == "Male" else 0
            
        elif key == "fbs" or key == "exang":
            # Use a select slider for fasting blood sugar to show Yes/No
            input_dict[key] = st.sidebar.select_slider(
                label,
                options=["No", "Yes"],  # Display labels
                value="No"  # Default value as No
            )
            # Map the selected label back to numerical values
            input_dict[key] = 1 if input_dict[key] == "Yes" else 0
            
        elif key in ["cp", "restecg", "slope", "ca", "thal"]:
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=int(data[key].min()),  # Ensure this is int
                max_value=int(data[key].max()),  # Ensure this is int
                value=int(data[key].mean()),  # Ensure this is int
                step=1  # Step to ensure integer selection
            )
        else:
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=float(data[key].min()),
                max_value=float(data[key].max()),
                value=float(data[key].mean())
            )
        
    return input_dict

def get_scaled_values(input_dict):
    data = pd.read_csv(r"C:\Users\calli\OneDrive\Desktop\Strands\Deep Learning\Heart Disease Classification\Data\heart.csv")
    predictors = data.drop(['target'], axis=1)
  
    scaled_dict = {}
    
    for key, value in input_dict.items():
        max_val = predictors[key].max()
        min_val = predictors[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
  
    return scaled_dict

# Function to generate the radar chart for visualizing feature data
def get_radar_chart(input_data):
  
      # Assuming you scale or preprocess input_data here
    input_data = get_scaled_values(input_data)
    # Updated feature categories for the radar chart (excluding sex, fbs, exang, and target)
    categories = ['Age', 'Chest Pain Type', 'Resting BP', 
                  'Cholesterol', 'Resting ECG', 'Max HR', 
                  'Oldpeak']

    fig = go.Figure()

    # Plot the "Mean Value" trace
    fig.add_trace(go.Scatterpolar(
          r=[
            input_data['age'], input_data['cp'], input_data['trestbps'], 
            input_data['chol'], input_data['restecg'], input_data['thalach'], 
            input_data['oldpeak']
          ],
          theta=categories,
          fill='toself',
          name='Heart Disease Features'
    ))

    # Set up radar chart layout
    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]  # Adjust range based on your feature scaling
        )),
    showlegend=True,
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the chart itself
    plot_bgcolor='rgba(0,0,0,0)'   # Transparent background for the plotting area
    )

    return fig

# Function to make predictions using the pre-trained model
def add_predictions(input_data):
    model = pickle.load(open("ModPkl/model.pkl", "rb"))  # Load the pre-trained model
    scaler = pickle.load(open("ModPkl/scaler.pkl", "rb"))  # Load the scaler for input scaling
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)  # Reshape the input data into a 2D array
    
    input_array_scaled = scaler.transform(input_array)  # Apply scaling
    
    prediction = model.predict(input_array_scaled)  # Get prediction from the model
    
    box_color = '#404040'  # Adjust the color as desired

    # Create a styled box for the prediction result
    st.markdown(
        f"""
        <div style="padding: 1rem; border-radius: 0.5rem; background-color: {box_color};">
            <h2 style="color: white;">Prognosis</h2>
            <p style="color: white;">The prediction result is:</p>
            <p style="color: white;">
                {"<span class='diagnosis malignant'>Heart Disease</span>" if prediction[0] == 1 else "<span class='diagnosis benign'>No Heart Disease</span>"}
            </p>
            <h3 style="color: white;">Probability</h3>
            <p style="color: white;">
                Positive: {model.predict_proba(input_array_scaled)[0][0]:.2f}%<br>
                Negative: {(model.predict_proba(input_array_scaled)[0][1])*100:.2f}%
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()