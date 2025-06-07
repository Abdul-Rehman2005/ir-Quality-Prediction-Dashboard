import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import date
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

@st.cache_data
def get_df():
    df=pd.read_csv("Air_Quality.csv")
    df['Date'] = pd.to_datetime(df['Date'])  
    df['Month'] = df['Date'].dt.month        
    df['Day'] = df['Date'].dt.day            
    df['Weekday'] = df['Date'].dt.weekday 
    df=df.drop('CO2',axis=1)
    df=df.drop('Date',axis=1)

    return df
df=get_df()

st.sidebar.title("Navigation")
st.sidebar.subheader("Model paramter")
option = st.sidebar.selectbox(
    'Choose City:',
    ('Brasilia','Cairo','Dubai','London','New York','Sydney')
)

st.sidebar.write(" ")

range_size = df['CO'].max()-df['CO'].min()
step = math.ceil(range_size / 20) 
step = round(step / 100) * 100  
input_1 = st.sidebar.slider(
    'Select a value for CO',
    min_value=int(df['CO'].min()),
    max_value=int(df['CO'].max()),
    value=int(df['CO'].mean()),
    step=int(step)
)
st.sidebar.write(" ")
 
input_2 = st.sidebar.slider(
    'Select a value for NO2',
    min_value=int(df['NO2'].min()),
    max_value=int(df['NO2'].max()),
    value=int(df['NO2'].mean()),
    step=5
)
st.sidebar.write(" ")

input_3 = st.sidebar.slider(
    'Select a value for SO2',
    min_value=int(df['SO2'].min()),
    max_value=int(df['SO2'].max()),
    value=int(df['SO2'].mean()),
    step=5
)
st.sidebar.write(" ")

input_4 = st.sidebar.slider(
    'Select a value for O3',
    min_value=int(df['O3'].min()),
    max_value=int(df['O3'].max()),
    value=int(df['O3'].mean()),
    step=10
)
st.sidebar.write(" ")

input_5 = st.sidebar.slider(
    'Select a value for PM2.5',
    min_value=int(df['PM2.5'].min()),
    max_value=int(df['PM2.5'].max()),
    value=int(df['PM2.5'].mean()),
    step=5
)
st.sidebar.write(" ")

input_6 = st.sidebar.slider(
    'Select a value for PM10',
    min_value=int(df['PM10'].min()),
    max_value=int(df['PM10'].max()),
    value=int(df['PM10'].mean()),
    step=10
)
st.sidebar.write(" ")


def get_date():
    selected_date = st.sidebar.date_input(
    "Select date in 2024",
    value=date(2024, 1, 1),
    min_value=date(2024, 1, 1),
    max_value=date(2024, 12, 31)
    )
    return selected_date.month,  selected_date.day,  selected_date.weekday()

month , day, weekday =get_date()
st.sidebar.write(' ')

st.sidebar.subheader("Graph paramter")
option1 = st.sidebar.selectbox(
    'Choose pollutant for graph:',
    ('CO', 'NO2',	'SO2',	'O3','PM2.5','PM10')
)

x_input=[input_1,input_2,input_3,input_4,input_5,input_6,month,day,weekday]

@st.cache_resource(show_spinner="Training model...")
def train_model(city,data):
    df=data[data['City']==city]
    y=df['AQI']
    x=df.drop(['AQI','City'],axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    
    pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('rf', RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        max_features=0.8,
        random_state=42
        ))
    ])

    pipeline.fit(x_train, y_train)
    return pipeline, x_test, y_test
model, x_test, y_test = train_model(option,df)
def predict_results(model,x_input, x_test, y_test):
    x_input = np.array(x_input).reshape(1,-1)
    y_pred = model.predict(x_input)
    y_result = model.predict(x_test)
    mse = mean_squared_error(y_test,y_result)
    r2 = r2_score(y_test,y_result)
    return y_pred,mse,r2
@st.cache_resource
def get_aqi_condition(aqi):
    if aqi <= 50:
        return "Good 👍"
    elif aqi <= 100:
        return "Moderate 😐"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups ⚠️"
    elif aqi <= 200:
        return "Unhealthy ❌"
    elif aqi <= 300:
        return "Very Unhealthy ☠️"
    else:
        return "Hazardous 🚨"

y_pred, mse, r2 = predict_results(model,x_input,x_test, y_test)

st.title(option + " AQI Forecast Tool")
st.write("")
with st.expander("🔍 Click to view full data table"):
    st.dataframe(df[df['City']==option])
st.write("")

st.subheader("🔧 Adjust Pollution Parameters in the Sidebar")
st.markdown("""
Use the sidebar to adjust pollutant concentrations like **PM2.5, NO2, CO, O3, city**, etc.
This will update the model's prediction for AQI in real time.

📈 You can also view how these values affect the AQI through interactive graphs.
""")
st.write(' ')

column1, column2 = st.columns(2)
with column1:
    st.metric("Predicted AQI", f"{y_pred[0]:.1f}")
with column2:
    st.metric("Air Quality ", get_aqi_condition(y_pred[0]))

st.write(' ')
col1, col2 = st.columns(2)
with col1:
    st.metric("Model R² Score", f"{r2:.2f}")
with col2:
    st.metric("Mean Squared Error", f"{mse:.2f}")
st.write("")




def create_interactive_plot(df, option, option1):
    data = df[df['City'] == option]
    monthly_avg = data.groupby('Month')[option1].mean().reset_index()
    fig = px.line(
        monthly_avg,
        x='Month',
        y=option1,
        markers=True,
        title=f"Monthly Average of {option1} in {option}",
        labels={'Month': 'Month', option1: f'Average {option1}'},
    )

    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        plot_bgcolor='#2c003e',      
        paper_bgcolor='#2c003e',     
        font=dict(color='white'),    
        title_x=0,
        title_font=dict(size=20),
        margin=dict(t=60, l=30, r=30, b=30)
    )

    fig.update_traces(line=dict(color='#1f77b4', width=2))

    return fig

st.write(" ")
st.plotly_chart(create_interactive_plot(df, option, option1), use_container_width=True)
st.write(" ")
st.write(" ")

def pollution_dashboard_row(df, city_name):
    city_data = df[df['City'] == city_name]
    if city_data.empty:
        st.warning(f"No data found for city: {city_name}")
        return
    latest_row = city_data.iloc[-1]  
    pollution_data = {
        "CO": latest_row["CO"],
        "NO2": latest_row["NO2"],
        "SO2": latest_row["SO2"],
        "O3": latest_row["O3"],
        "PM2.5": latest_row["PM2.5"],
        "PM10": latest_row["PM10"]
    }
    df_pollutants = pd.DataFrame(pollution_data.items(), columns=["Pollutant", "Concentration"])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        specs=[[{"type": "bar"}, {"type": "domain"}]],
        subplot_titles=(f"{city_name} Pollutant Concentrations", "Contribution Share")
    )
    fig.add_trace(
        go.Bar(
            x=df_pollutants["Concentration"],
            y=df_pollutants["Pollutant"],
            orientation='h',
            marker_color=colors,
            name="Concentration"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Pie(
            labels=df_pollutants["Pollutant"],
            values=df_pollutants["Concentration"],
            marker_colors=colors,
            hole=0.3
        ),
        row=1, col=2
    )
    fig.update_layout(
        height=450,
        paper_bgcolor='#2e003e',
        plot_bgcolor='#2e003e',
        font_color='white',
        showlegend=False,
        title_text="Pollution Breakdown Overview",
        margin=dict(t=50, l=30, r=30)
    )
    st.plotly_chart(fig, use_container_width=True)

pollution_dashboard_row(df, option)


#st.dataframe(df)
print("done")