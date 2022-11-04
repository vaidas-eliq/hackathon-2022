import os
import zipfile

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor


st.set_page_config(
    page_title="Home Improvement Analysis",
    page_icon="🏠",
    layout="wide",
)
st.header("🏠 Home Improvement Analysis")

DATA_PWD = os.getenv("ELIQ_DATA_PWD", default="").encode()
if DATA_PWD.decode() == "":
    st.error("NO PASSWORD FOR DATA IS PROVIDED")
    st.stop()
with zipfile.ZipFile("./data/energy-data.zip") as data_zip:
    with data_zip.open("energy_hp_final.csv", pwd=DATA_PWD) as data_file:
        energy_hp = pd.read_csv(data_file)

house_type_dummies = pd.get_dummies(energy_hp["house_type"], prefix="house_type")
heating_type_dummies = pd.get_dummies(energy_hp["heating_type_primary"], prefix="heating_type")
hotwater_type_dummies = pd.get_dummies(energy_hp["hotwater_type"], prefix="hotwater_type")
numeric_features = energy_hp[["bedrooms", "persons", "electric_cars"]]
locations_features = energy_hp[["location_id"]].join(numeric_features).join(house_type_dummies).join(heating_type_dummies).join(hotwater_type_dummies)
features = locations_features.drop(columns="location_id")
target = energy_hp[["yearly-consumption"]]

PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
models = joblib.load("./models/percentile_models.sav")

location_ids = list(energy_hp["location_id"].unique())
location = st.selectbox("Select location", options=location_ids)
st.subheader(f"Location {location} information:")
col1, col2 = st.columns(2)
with col1:
    selected_locations_features = locations_features[locations_features["location_id"] == location].drop(columns="location_id")
    selected_location_hp = energy_hp[energy_hp["location_id"] == location].reset_index()[["location_id", "house_type", "heating_type_primary", "hotwater_type", "bedrooms", "persons", "electric_cars"]]

    selected_location_house_type = " ".join([word.capitalize() for word in selected_location_hp.loc[0, "house_type"].split("_")])
    selected_location_bedrooms = int(selected_location_hp.loc[0, "bedrooms"])
    selected_location_persons = int(selected_location_hp.loc[0, "persons"])
    selected_location_heating_type = " ".join([word.capitalize() for word in selected_location_hp.loc[0, "heating_type_primary"].split("_")])
    selected_location_hot_water_type = " ".join([word.capitalize() for word in selected_location_hp.loc[0, "hotwater_type"].split("_")])
    selected_location_electric_cars = int(selected_location_hp.loc[0, "electric_cars"])

    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")

    st.markdown(f"""
    | Attribute | Value |
    | :--------- | :-----: |
    | House type | {selected_location_house_type} |
    | Heating type | {selected_location_heating_type} |
    | Hot water type | {selected_location_hot_water_type} |
    | Number of bedrooms | {selected_location_bedrooms} |
    | Number of persons | {selected_location_persons} |
    | Number of electric cars | {selected_location_electric_cars} |
    """)

with col2:
    actual_value = energy_hp[energy_hp["location_id"]==location]["yearly-consumption"].iloc[0]
    predictions = []
    for model in PERCENTILES:
        predictions.append(models[model].predict(selected_locations_features)[0] / 1000.)

    # Find the current precentile
    actual_percentile = 100
    actual_value_kwh = actual_value / 1000.
    for percentile, prediction in zip(PERCENTILES, predictions):
        if prediction > actual_value_kwh:
            actual_percentile = actual_value_kwh / prediction * percentile


    fig, ax = plt.subplots()
    fig.set_size_inches(6, 4)
    _ = ax.plot(PERCENTILES, predictions, color="purple")
    ax.set_ylabel("kWh")
    ax.set_xlabel("Percentile")
    _ = plt.title("Yearly energy consumption comparison to simiarl homes")
    _ = plt.vlines(actual_percentile, ymin=actual_value_kwh-500, ymax=actual_value_kwh+500, color="red")
    st.pyplot(fig)

st.subheader(f"Explore heating alternatives")
heating_types = list(energy_hp["heating_type_primary"].unique())
current_heating_type_index = heating_types.index(selected_location_hp.loc[0, "heating_type_primary"])
new_heating = st.selectbox(
    "Select heating type",
    options=heating_types, 
    index=current_heating_type_index, 
    format_func=lambda s: " ".join([word.capitalize() for word in s.split("_")])
)
selected_locations_index = selected_locations_features.index[0]
selected_locations_features.loc[selected_locations_index, f"heating_type_{selected_location_hp.loc[0, 'heating_type_primary']}"] = 0.
selected_locations_features.loc[selected_locations_index, f"heating_type_{new_heating}"] = 1.

new_heating_predictions = []
for model in PERCENTILES:
    new_heating_predictions.append(models[model].predict(selected_locations_features)[0] / 1000.)

average_change = 0.
for old, new in zip(predictions, new_heating_predictions):
    average_change += new / old - 1
average_change = average_change / 9. * 100.
new_estimated_consumption = (actual_value_kwh * (1+average_change/100))
new_heating_type_formatted = ' '.join([w.capitalize() for w in new_heating.split('_')])
if new_heating_type_formatted != selected_location_heating_type:
    col3, col4 = st.columns(2)
    with col3:
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")

        st.markdown(f"""
        | Attribute | Value |
        | :--------- | :-----: |
        | Original Heating Type| {selected_location_heating_type} |
        | New Heating type | {new_heating_type_formatted} |
        | Average Change | {average_change:.2f} % |
        | Original Consumption | {actual_value_kwh:,.0f} kWh |
        | New Estimated Consumption | {new_estimated_consumption:,.0f} kWh |
        """)
    with col4:
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 4)
        _ = ax.plot(PERCENTILES, predictions, color="purple", label=f"Original - {selected_location_heating_type}")
        _ = ax.plot(PERCENTILES, new_heating_predictions, color="green", label=f"New - {' '.join([w.capitalize() for w in new_heating.split('_')])}")
        ax.set_ylabel("kWh")
        ax.set_xlabel("Percentile")
        _ = plt.title("Orignal vs. New Heating Type")
        _ = plt.legend()
        st.pyplot(fig)
