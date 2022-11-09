import os
import zipfile

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

GAS_OIL = [
    "Gas Boiler With Radiators",
    "Oil Boiler With Radiators",
    "Solid Fuel",
]
ELECTRIC = [
     "Plug In Electric Heaters",
    "Night Storage Heaters",
    "Modern Slimline Night Storage Heaters",
    "Heat Pump",
    "Electric Boiler",
    "Floor Heat",
]


st.set_page_config(
    page_title="Similar Homes Explorer",
    page_icon="🏠",
    layout="wide",
)
st.header("🏠 Similar Homes Explorer")

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

col_location, col_price = st.columns(2)
with col_location:
    location_ids = list(energy_hp["location_id"].unique())
    location = st.selectbox("Select location", options=location_ids)
    st.write("Suggestion 1: 1839899 - Oil Boiler With Radiators")
    st.write("Suggestion 2: 1850228 - Gas Boiler With Radiators")

with col_price:
    energy_price = st.number_input("Your average electricity price per kWh", value=0.0)

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
    | House Type | {selected_location_house_type} |
    | Heating Type | {selected_location_heating_type} |
    | Hot Water Type | {selected_location_hot_water_type} |
    | Number of Bedrooms | {selected_location_bedrooms} |
    | Number of Persons | {selected_location_persons} |
    | Number of Electric Cars | {selected_location_electric_cars} |
    """)

with col2:
    actual_value = energy_hp[energy_hp["location_id"]==location]["yearly-consumption"].iloc[0]
    predictions = []
    for model in PERCENTILES:
        predictions.append(models[model].predict(selected_locations_features)[0] / 1000.)

    # Find the current precentile
    actual_percentile = 100
    actual_value_kwh = actual_value / 1000.
    for idx, percentile in enumerate(PERCENTILES):
        prediction = predictions[idx]
        if prediction > actual_value_kwh:
            if idx > 0:
                previous_percentile = PERCENTILES[idx-1]
                previous_prediction = predictions[idx-1]
                point = (actual_value_kwh - previous_prediction) / (prediction - previous_prediction)
                actual_percentile = previous_percentile + ((percentile - previous_percentile) * point)
            else:
                actual_percentile = percentile
            break

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 4)
    _ = ax.plot(PERCENTILES, predictions, color="purple")
    ax.set_ylabel("kWh")
    ax.set_xlabel("Percentile")
    _ = plt.title("Yearly energy consumption comparison to similar homes")
    _ = plt.vlines(actual_percentile, ymin=actual_value_kwh-500, ymax=actual_value_kwh+500, color="red")
    st.pyplot(fig)

st.subheader("Comparison with (other) electric heating options")
if energy_price > 0.0:
    do_comparison_money = st.checkbox("Compare Heating Aleternatives by Costs")
if selected_location_heating_type in ELECTRIC:
    
    other_electric = [heating_type for heating_type in ELECTRIC if heating_type != selected_location_heating_type]
    other_electric_predictions = {}
    previous_heating_type = selected_location_hp.loc[0, 'heating_type_primary']
    selected_locations_index = selected_locations_features.index[0]
    for heating_type in other_electric:
        heating_alternative = "_".join([heating_type_part.lower() for heating_type_part in heating_type.split(" ")])
        selected_locations_features.loc[selected_locations_index, f"heating_type_{previous_heating_type}"] = 0.
        selected_locations_features.loc[selected_locations_index, f"heating_type_{heating_alternative}"] = 1.
        other_electric_predictions[heating_type] = []
        for model in PERCENTILES:
            other_electric_predictions[heating_type].append(models[model].predict(selected_locations_features)[0] / 1000.)
        
        previous_heating_type = heating_alternative
    
    # Calculate average differences across the curves
    heating_alternative_diffs = {}
    for heating_type in other_electric:
        heating_alternative_diffs[heating_type] = float(np.mean((np.array(other_electric_predictions[heating_type]) / np.array(predictions) - 1.)))
    
    heating_alternative_values = []
    formatted_diffs = []
    for heating_type in other_electric:
        diff = heating_alternative_diffs[heating_type]
        heating_alternative_values.append((1. + diff) * actual_value_kwh)
        formatted_diffs.append(f"{diff * 100.0:.2f} %")
    
    if (energy_price > 0.0) and do_comparison_money:

        actual_value_money = actual_value_kwh * energy_price
        heating_alternative_values_money = [alternative * energy_price for alternative in heating_alternative_values]
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 4)

        _ = ax.bar(other_electric, heating_alternative_values_money, color="purple")
        _ = plt.hlines(actual_value_money, -0.3, 4.3, linestyles="-.", label=selected_location_heating_type, color="green")
        ax.set_ylabel("Pounds")
        ax.set_xlabel("Heating Type")
        _ = plt.title("Energy consumption comparison for different heating types")
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.legend(loc="lower left")
        ax.bar_label(ax.containers[0], formatted_diffs)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 4)

        _ = ax.bar(other_electric, heating_alternative_values, color="purple")
        _ = plt.hlines(actual_value_kwh, -0.3, 4.3, linestyles="-.", label=selected_location_heating_type, color="green")
        ax.set_ylabel("kWh")
        ax.set_xlabel("Heating Type")
        _ = plt.title("Energy consumption comparison for different heating types")
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.legend(loc="lower left")
        ax.bar_label(ax.containers[0], formatted_diffs)
        st.pyplot(fig)


if selected_location_heating_type in GAS_OIL:
    electric_predictions = {}
    previous_heating_type = selected_location_hp.loc[0, 'heating_type_primary']
    selected_locations_index = selected_locations_features.index[0]
    for heating_type in ELECTRIC:
        heating_alternative = "_".join([heating_type_part.lower() for heating_type_part in heating_type.split(" ")])
        selected_locations_features.loc[selected_locations_index, f"heating_type_{previous_heating_type}"] = 0.
        selected_locations_features.loc[selected_locations_index, f"heating_type_{heating_alternative}"] = 1.
        electric_predictions[heating_type] = []
        for model in PERCENTILES:
            electric_predictions[heating_type].append(models[model].predict(selected_locations_features)[0] / 1000.)
        
        previous_heating_type = heating_alternative
    
    heating_alternative_diffs = {}
    for heating_type in ELECTRIC:
        heating_alternative_diffs[heating_type] = float(np.mean((np.array(electric_predictions[heating_type]) / np.array(predictions) - 1.)))
    
    heating_alternative_values = []
    formatted_diffs = []
    min_idx = None
    for idx, heating_type in enumerate(ELECTRIC):
        diff = heating_alternative_diffs[heating_type]
        heating_alternative_values.append(diff * actual_value_kwh)
        formatted_diffs.append("")
        if min_idx is None:
            min_idx = idx
            min_value = diff
        else:
            if diff < min_value:
                min_idx = idx
    
    formatted_diffs[min_idx] = "*"
    
    if (energy_price > 0.0) and do_comparison_money:

        heating_alternative_values_money = [alternative * energy_price for alternative in heating_alternative_values]

        fig, ax = plt.subplots()
        fig.set_size_inches(6, 4)

        _ = ax.bar(ELECTRIC, heating_alternative_values, color="purple")
        ax.set_ylabel("Pounds")
        ax.set_xlabel("Heating Type")
        _ = plt.title("Additional Energy to Heat Location with Electricity")
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
        ax.bar_label(ax.containers[0], formatted_diffs)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 4)

        _ = ax.bar(ELECTRIC, heating_alternative_values, color="purple")
        ax.set_ylabel("kWh")
        ax.set_xlabel("Heating Type")
        _ = plt.title("Additional Energy to Heat Location with Electricity")
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
        ax.bar_label(ax.containers[0], formatted_diffs)
        st.pyplot(fig)

with st.expander("Explore alternative heating methods percentile curves"):
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

    if (selected_location_heating_type.strip() in GAS_OIL) and (new_heating_type_formatted.strip() in ELECTRIC):
        st.subheader("Impact on electricity bill from the switch")
        if energy_price > 0.0:
            impact_on_bill = int((new_estimated_consumption - actual_value_kwh) * energy_price)
            if impact_on_bill > 0.:
                st.markdown(f"You would have paid additional **{impact_on_bill}** Pounds for the previous year.")
            else:
                st.markdown(f"You would have saved **{impact_on_bill}** Pounds for the previous year.")
