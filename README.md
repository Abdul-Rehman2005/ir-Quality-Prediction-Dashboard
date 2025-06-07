# ir-Quality-Prediction-Dashboard
This project is a user-friendly web dashboard built using Streamlit that allows users to explore and predict Air Quality Index (AQI) based on pollution data. It uses a Random Forest Regression model to estimate AQI from pollutant concentrations, and provides helpful visualizations to understand pollution trends over time for different cities.

Overview
The dashboard is designed for environmental monitoring and educational purposes. Users can select a city, adjust pollution concentration levels (such as CO, NO₂, PM2.5, etc.), and get an AQI prediction based on a machine learning model trained with historical data.

The interface also includes graphs to visualize how pollutants behave over time, and model evaluation metrics like Mean Squared Error (MSE) and R² Score.

Features
Predict AQI using a Random Forest regression model

Input pollutant concentrations and get instant AQI predictions

View model accuracy metrics like R² score and mean squared error

Explore monthly pollutant trends for any city with interactive graphs

See pollutant contribution using bar and pie charts

Choose whether to view the raw dataset

Clean and styled interface with modern color themes

Technologies Used
Python – Core programming language

Pandas – Data manipulation

Scikit-learn – Machine learning model and pipeline

Streamlit – Web application framework

Plotly – Interactive visualizations

Matplotlib – Static graphing (used where necessary)

How It Works
User selects a city from the sidebar.

Pollutant levels can be adjusted using sliders.

The trained model takes these inputs and predicts the AQI.

The dashboard displays:

Predicted AQI value

Health condition message based on the AQI

Line graph of monthly average of any selected pollutant

Model evaluation scores (MSE and R²)

Pollutant contribution in the form of bar and pie charts

Optional: Users can expand a section to view the actual data being used.

Model Information
Model type: Random Forest Regressor

Pipeline includes: Missing value imputation and model training

Evaluation: Accuracy is measured using R² Score and Mean Squared Error

The model performs well with a typical R² of around 0.99 and low error (MSE ~1.7), making it highly reliable for predictions on known data patterns.

License
This project is released under the MIT License. You can use, modify, and distribute it as needed.

Contact
If you have questions, suggestions, or want to collaborate, feel free to open an issue or reach out via GitHub
