import streamlit as st
import pandas as pd
import pickle
import os

# Load files
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")

st.title("🚗 Car Price Prediction")
st.image('imag1.jpg' , caption="car image " , use_container_width=True)
st.write("Enter the car details below.")

# ---------- Inputs ----------

levy = st.number_input("Levy", value=1000.0)

manufacturer = st.selectbox("Manufacturer",
[        'LEXUS',     'CHEVROLET',         'HONDA',          'FORD',
       'HYUNDAI',        'TOYOTA', 'MERCEDES-BENZ',          'OPEL',
       'PORSCHE',           'BMW',          'JEEP',    'VOLKSWAGEN',
          'AUDI',       'RENAULT',        'NISSAN',        'SUBARU',
        'DAEWOO',           'KIA',    'MITSUBISHI',     'SSANGYONG',
         'MAZDA',           'GMC',          'FIAT',      'INFINITI',
    'ALFA ROMEO',        'SUZUKI',         'ACURA',       'LINCOLN',
           'VAZ',           'GAZ',       'CITROEN',    'LAND ROVER',
          'MINI',         'DODGE',      'CHRYSLER',        'JAGUAR',
         'ISUZU',         'SKODA',      'DAIHATSU',         'BUICK',
         'TESLA',      'CADILLAC',       'PEUGEOT',       'BENTLEY',
         'VOLVO',          'სხვა',         'HAVAL',        'HUMMER',
         'SCION',           'UAZ',       'MERCURY',           'ZAZ',
         'ROVER',          'SEAT',        'LANCIA',      'MOSKVICH',
      'MASERATI',       'FERRARI',          'SAAB',   'LAMBORGHINI',
   'ROLLS-ROYCE',       'PONTIAC',        'SATURN',  'ASTON MARTIN',
     'GREATWALL']
)

model_name = st.selectbox("Model", [    'RX 450','Equinox','FIT','Escape','Santa FE','Prius','Sonata','Camry','RX 350','E 350','C 240 W 203','Vito Extralong','E 500 AVG','530 i','FIT LX','Every Landy NISSAN SEREN','CL 600','E 230 124','RX 450 F SPORT','Prius C aqua'])


prod_year = st.number_input("Production Year", 1950, 2025, 2015)

category = st.selectbox("Category",['Jeep',   'Hatchback',       'Sedan',    'Microbus', 'Goods wagon',
   'Universal',       'Coupe',     'Minivan',   'Cabriolet',   'Limousine',
      'Pickup'] )

leather = st.text_input("Leather Interior (Yes/No)", "Yes")

fuel =st.selectbox("Fuel Type",['Hybrid', 'Petrol', 'Diesel', 'CNG', 'Plug-in Hybrid', 'LPG', 'Hydrogen'] )

engine = st.number_input("Engine Volume", value=2.0)

mileage = st.number_input("Mileage", value=50000)

cylinders = st.number_input("Cylinders", value=4)

gearbox =st.selectbox("Gear Box Type", ['Automatic', 'Tiptronic', 'Variator', 'Manual'])

drive = st.selectbox("Drive Wheels", ['4x4', 'Front', 'Rear'])

color = st.selectbox("Color", ['Silver',         'Black',         'White',          'Grey',
          'Blue',         'Green',           'Red',      'Sky blue',
        'Orange',        'Yellow',         'Brown',        'Golden',
         'Beige', 'Carnelian red',        'Purple',          'Pink'])

airbags = st.number_input("Airbags", value=4)

# ---------- Predict ----------

if st.button("🔍 Predict Price"):

    new_data = pd.DataFrame({
        "Levy":[levy],
        "Manufacturer":[manufacturer],
        "Model":[model_name],
        "Prod. year":[prod_year],
        "Category":[category],
        "Leather interior":[leather],
        "Fuel type":[fuel],
        "Engine volume":[engine],
        "Mileage":[mileage],
        "Cylinders":[cylinders],
        "Gear box type":[gearbox],
        "Drive wheels":[drive],
        "Color":[color],
        "Airbags":[airbags]
    })

    # Encode categorical columns
    cat_cols = [
        "Manufacturer",
        "Model",
        "Category",
        "Leather interior",
        "Fuel type",
        "Gear box type",
        "Drive wheels",
        "Color"
    ]

    for col in cat_cols:
        new_data[col] = encoder.fit_transform(new_data[col].astype(str))

    # Scale
    new_data_scaled = scaler.transform(new_data)

    # Predict
    prediction = model.predict(new_data_scaled)

    st.success(f"💰 Predicted Car Price: ₹ {prediction[0]:,.2f}")

port = int(os.environ.get("PORT", 8501))
