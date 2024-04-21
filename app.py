import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st
import sklearn
import plotly.graph_objects as go
import plotly.express as px
import joblib
import folium
import statsmodels
from sklearn.model_selection import train_test_split


# Load the dataset
df_ML = pd.read_csv("ML_data.csv")

# Set page title and favicon
st.set_page_config(page_title="NPRI Project", page_icon="📊")

def dashboard():
    st.image('Picture1.png', width=350)

    # Abstract section
    st.subheader("Welcome to Our NPRI Project: Predicting releases")
    st.subheader("💡 Abstract:")
    inspiration = '''
    Our NPRI (National Pollutant Release Inventory) project in Canada aimed to address the challenge of predicting releases for oil and gas sector in 5-10 years based on economic factors. Using a dataset provided by NPRI, which included features like NpriID, company name, substance name, and quantity, we encountered the issue that the number of reporting facilities can fluctuate depending on economic conditions.

**👨🏻‍💻Here's the adventure we embarked on:**

1. **Data Discovery:** We delved into the dataset from 2014 to 2022, uncovering trends in pollution and economics.
2. **Data Enhancement:** To tackle economic fluctuations, we enriched our dataset with crucial indicators like prices and growth rates.
3. **Feature Engineering:** We refined the dataset, eliminating unnecessary elements and identifying correlations between variables.
4. **Machine Learning Magic:** Leveraging advanced machine learning techniques, we utilized a linear regression model to forecast pollution quantities for the next five years.
5. **User-Friendly Interface:** Lastly, we developed a user-friendly application to visualize predicted pollution trends, ensuring accessibility for all stakeholders.
    '''
    st.markdown(f'<div style="background-color:#c6ffb3; padding: 15px; border-radius: 10px;"><p style="color:#000000 ;font-size:16px;">{inspiration}</p></div>', unsafe_allow_html=True)
# Set up the Streamlit app with a custom background color and text colors
st.markdown(
    """
    <style>
    body {
        background-color: #d4f4dd; /* Light green */
        color: #333333; /* Dark blue */
    }
    .feature-box {
        background-color: #c7ecee; /* Light blue */
        padding: 10px ;
        border-radius: 10px ;
        box-shadow: 0px 0px 10px 0px rgba(0,0,0.1,0.9);
        margin-right: 10px;
        margin-bottom: 10px;
        display: inline-block;
        width: 150px; /* Adjust the width of the feature boxes */
        text-align: center; /* Center align text */
    }
    .feature-box:nth-child(even) {
        background-color: #a9cce3; /* Light blue for alternate boxes */
    }
    .title-text {
        font-size: 18px;
        color: #004c6d; /* Dark blue */
        font-weight: bold; /* Make title text bold */
    }
    .subheader-text {
        font-size: 16px;
        color: #666666; /* Dark blue */
    }
    .selected-feature {
        font-weight: bold; /* Make selected feature text bold */
    }
    .predicted-quantity {
        font-weight: bold; /* Make predicted quantity text bold */
        color: #000000; /* Black text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

def machine_learning_modeling():
    # Header
    st.title("NPRI ML Model")

    # Sidebar
    st.sidebar.header('Select Features')

    # Select box for categorical features
    st.subheader('Selected Features')

    def user_input_features():
        company_name_options = ['All'] + list(df_ML['Company name'].unique())
        default_company_name = 'Canadian Natural Resources Limited'  # Default company name
        company_name = st.sidebar.selectbox('Company Name', company_name_options, index=company_name_options.index(default_company_name))
        province = ""
        if company_name != 'All':
            province = df_ML[df_ML['Company name'] == company_name]['Province'].iloc[0]
        substance_name = st.sidebar.selectbox('Substance Name', df_ML['Substance name'].unique())
        number_of_employees = st.sidebar.number_input('Number of Employees', min_value=1, max_value=5000, value=10, step=1)
        growth_input = st.sidebar.radio('Growth', ('up', 'down'))
        Price_input = st.sidebar.radio('Price', ('up', 'down'))

        user_input_data = {'Company Name': company_name,
                           'Province': province,
                           'Substance Name': substance_name,
                           'Number of Employees': number_of_employees,
                           'Growth': growth_input,
                           'Price': Price_input}

        features = pd.DataFrame(user_input_data, index=[0])

        return features

    df_user_input = user_input_features()

    st.write(df_user_input.style.set_properties(**{'font-weight': 'bold'}))

    # Placeholder for displaying predicted quantities for 2019 to 2027
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("<p class='title-text'><b>Predicted Quantities For 2019 to 2027:</b></p>", unsafe_allow_html=True)

    # Filter data for the selected substance name and company name
    selected_substance_name = df_user_input['Substance Name'].iloc[0]
    selected_company_name = df_user_input['Company Name'].iloc[0]

    if selected_company_name == 'All':
        filtered_data = df_ML[df_ML['Substance name'] == selected_substance_name]
        if not filtered_data.empty:
            # Train a Linear Regression model using data from 2014 to 2022
            model = LinearRegression()
            model.fit(df_ML[['Last year quantity', 'Two years ago quantity', 'Three years ago quantity',
                             'Number of employees', 'Growth', 'Price']][(df_ML['NPRI_Report_ReportYear'] >= 2014) & (df_ML['NPRI_Report_ReportYear'] < 2023)],
                      df_ML['Current year quantity'][(df_ML['NPRI_Report_ReportYear'] >= 2014) & (df_ML['NPRI_Report_ReportYear'] < 2023)])

            # Initialize input data for prediction
            input_data = {
                'Last year quantity': filtered_data['Last year quantity'].mean(),
                'Two years ago quantity': filtered_data['Two years ago quantity'].mean(),
                'Three years ago quantity': filtered_data['Three years ago quantity'].mean(),
                'Number of employees': df_user_input['Number of Employees'].iloc[0],
                'Growth': 1 if df_user_input['Growth'].iloc[0] == 'up' else 0,
                'Price': 1 if df_user_input['Price'].iloc[0] == 'up' else 0,
            }

            # Predict quantities for the years 2019 to 2027
            predicted_quantities = []
            for year in range(2019, 2028):
                if year < 2023:
                    # Fetch current year quantity from the original dataset
                    quantity = filtered_data['Current year quantity'][(filtered_data['NPRI_Report_ReportYear'] == year)]
                    predicted_quantities.append(quantity.iloc[0] if not quantity.empty else np.nan)
                else:
                    # Predict quantity using the machine learning model
                    prediction = model.predict(np.array([list(input_data.values())]))
                    predicted_quantities.append(prediction[0])
                    # Update input data for the next prediction
                    input_data['Three years ago quantity'] = input_data['Two years ago quantity']
                    input_data['Two years ago quantity'] = input_data['Last year quantity']
                    input_data['Last year quantity'] = prediction[0]

            # Plot the trend graph with Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(2019, 2028)), y=predicted_quantities,
                                     mode='lines+markers',
                                     name=f'Predicted Quantity',
                                     line=dict(color='rgb(31, 119, 180)', width=2),
                                     marker=dict(color='rgb(31, 119, 180)', size=8, line=dict(color='white', width=2))))
            fig.update_layout(title=f'Trend of {selected_substance_name} Quantity from 2019 to 2027',
                              xaxis_title='Year',
                              yaxis_title='Quantity (tonnes)',
                              template='plotly_white',
                              font=dict(family="Arial, sans-serif", size=12, color="black"),
                              margin=dict(l=50, r=50, t=50, b=50),
                              xaxis=dict(tickmode='linear', tick0=2019, dtick=1),  # Set tick mode to linear and start from 2019 with step 1
                              xaxis_range=[2019, 2027])  # Set the x-axis range from 2019 to 2027

            # Add annotations for quantity values
            for i, txt in enumerate(predicted_quantities):
                fig.add_annotation(x=range(2019, 2028)[i], y=predicted_quantities[i],
                                   text=f'{txt:.2f}',
                                   showarrow=False,
                                   yshift=10)

            st.plotly_chart(fig)

        else:
            st.write(f"No data available for {selected_substance_name} for {selected_company_name} for previous years.")

    else:
        filtered_data = df_ML[(df_ML['Substance name'] == selected_substance_name) & (df_ML['Company name'] == selected_company_name)]
        if not filtered_data.empty:
            # Train a Linear Regression model using data from 2014 to 2022
            model = LinearRegression()
            model.fit(filtered_data[['Last year quantity', 'Two years ago quantity', 'Three years ago quantity',
                                     'Number of employees', 'Growth', 'Price']][(filtered_data['NPRI_Report_ReportYear'] >= 2014) & (filtered_data['NPRI_Report_ReportYear'] < 2023)],
                      filtered_data['Current year quantity'][(filtered_data['NPRI_Report_ReportYear'] >= 2014) & (filtered_data['NPRI_Report_ReportYear'] < 2023)])

            # Initialize input data for prediction
            input_data = {
                'Last year quantity': [filtered_data['Current year quantity'].iloc[0]],
                'Two years ago quantity': [filtered_data['Last year quantity'].iloc[0]],
                'Three years ago quantity': [filtered_data['Two years ago quantity'].iloc[0]],
                'Number of employees': [df_user_input['Number of Employees'].iloc[0]],
                'Growth': [1 if df_user_input['Growth'].iloc[0] == 'up' else 0],
                'Price': [1 if df_user_input['Price'].iloc[0] == 'up' else 0],
            }

            # Predict quantities for the years 2019 to 2027
            predicted_quantities = []
            for year in range(2019, 2028):
                if year < 2023:
                    # Fetch current year quantity from the original dataset
                    quantity = filtered_data['Current year quantity'][(filtered_data['NPRI_Report_ReportYear'] == year)]
                    predicted_quantities.append(quantity.iloc[0] if not quantity.empty else np.nan)
                else:
                    # Predict quantity using the machine learning model
                    prediction = model.predict(pd.DataFrame(input_data))
                    predicted_quantities.append(prediction[0])
                    # Update input data for the next prediction
                    input_data['Three years ago quantity'] = input_data['Two years ago quantity']
                    input_data['Two years ago quantity'] = input_data['Last year quantity']
                    input_data['Last year quantity'] = [prediction[0]]

            # Plot the trend graph with Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(2019, 2028)), y=predicted_quantities,
                                     mode='lines+markers',
                                     name=f'Predicted Quantity',
                                     line=dict(color='rgb(31, 119, 180)', width=2),
                                     marker=dict(color='rgb(31, 119, 180)', size=8, line=dict(color='white', width=2))))
            fig.update_layout(title=f'Trend of {selected_substance_name} Quantity for {selected_company_name} from 2019 to 2027',
                              xaxis_title='Year',
                              yaxis_title='Quantity (tonnes)',
                              template='plotly_white',
                              font=dict(family="Arial, sans-serif", size=12, color="black"),
                              margin=dict(l=50, r=50, t=50, b=50),
                              xaxis=dict(tickmode='linear', tick0=2019, dtick=1),  # Set tick mode to linear and start from 2019 with step 1
                              xaxis_range=[2019, 2027])  # Set the x-axis range from 2019 to 2027

            # Add annotations for quantity values
            for i, txt in enumerate(predicted_quantities):
                fig.add_annotation(x=range(2019, 2028)[i], y=predicted_quantities[i],
                                   text=f'{txt:.2f}',
                                   showarrow=False,
                                   yshift=10)

            st.plotly_chart(fig)

        else:
            st.write(f"No data available for {selected_substance_name} for {selected_company_name} for previous years.")

def Chatbot():
    # Chat Box
  st.subheader("🌟Welcome to our chatbot Assistant for NPRI Project🌟")  
  st.subheader("Have questions? Ask our Assistant!")
  chatbot_url = "https://hf.co/chat/assistant/661b432f5693cfc26defd2c3"
  st.markdown(f'<iframe src="{chatbot_url}" width="500" height="500"></iframe>', unsafe_allow_html=True)

def main():
    st.sidebar.title("NPRI App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "ML Modeling","Chatbot Asssitant Agent"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "Chatbot Asssitant Agent":
        Chatbot()


if __name__ == "__main__":
    main()
