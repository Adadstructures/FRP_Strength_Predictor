import numpy as np
import pickle
import streamlit as st

# Function to load the model
def load_model(file_path):
    return pickle.load(open(file_path, 'rb'))

# Function to make strain predictions
def strain_prediction(model, input_data):
    input_data = np.asarray(input_data).reshape(1, -1)
    return model.predict(input_data)[0]

# Function to make strength predictions
def strength_prediction(model, input_data):
    input_data = np.asarray(input_data).reshape(1, -1)
    return model.predict(input_data)[0]

# Function to calculate parameters
def calculate_parameters(Unconfined_Strength, Fibre_Modulus, FRP_Overall_Thickness, Diameter):
    Unconfined_Strain = 9.37 * 0.0001 * (Unconfined_Strength ** 0.25)
    Concrete_Modulus = 4730.0 * (Unconfined_Strength ** 0.5)
    
    # Check for zero division before performing the division
    if Diameter != 0:
        Confinement_Stiffness = 2.0 * Fibre_Modulus * FRP_Overall_Thickness / Diameter
    else:
        Confinement_Stiffness = 0
    
    Stiffness_Ratio = 0 if Unconfined_Strength == 0 else Confinement_Stiffness / (Unconfined_Strength / Unconfined_Strain)
    return Unconfined_Strain, Concrete_Modulus, Confinement_Stiffness, Stiffness_Ratio

# Main function for Streamlit app
def main():
    # Set page title and header styling
    st.title('FRP-Confined Concrete Strength Predictor')
    st.markdown('## Input Parameters')

    # Load Models
    loaded_xgb1 = load_model('C:/Users/Temitope/trained_xgb_strain.sav')
    loaded_xgb2 = load_model('C:/Users/Temitope/trained_catb_strength.sav')

    # Input Parameters
    col1, col2 = st.columns(2)

    with col1:
        Diameter = st.number_input('Diameter of Cylinder')

    with col2:
        Height = st.number_input('Height of Cylinder')

    # Group related input parameters
    with st.expander("Concrete Properties"):
        Unconfined_Strength = st.number_input('Unconfined Strength of Concrete')
        Concrete_Modulus = 4730.0 * (Unconfined_Strength ** 0.5)

    with st.expander("FRP Properties"):
        Fibre_Modulus = st.number_input('Modulus of FRP')
        FRP_Overall_Thickness = st.number_input('Overall Thickness of FRP')

    # Calculated Parameters
    st.header('Calculated Parameters')

    Unconfined_Strain, Concrete_Modulus, Confinement_Stiffness, Stiffness_Ratio = calculate_parameters(
        Unconfined_Strength, Fibre_Modulus, FRP_Overall_Thickness, Diameter
    )

    Confined_Strain = strain_prediction(
        loaded_xgb1, [Diameter, Height, Unconfined_Strength, Unconfined_Strain, Fibre_Modulus,
                      FRP_Overall_Thickness, Concrete_Modulus, Confinement_Stiffness, Stiffness_Ratio]
    )

    Rupture_Strain = 0.583 * Confined_Strain
    Confinement_Stress = Confinement_Stiffness * Rupture_Strain
    Strain_Ratio = Rupture_Strain / Unconfined_Strain if Unconfined_Strain != 0 else 0

    # Button for prediction
    if st.button('Predict Confined Strength', key='predict_button'):
        try:
            FRP_Strength_Prediction = strength_prediction(
                loaded_xgb2, [Diameter, Height, Unconfined_Strength, Unconfined_Strain, Fibre_Modulus,
                              FRP_Overall_Thickness, Confined_Strain, Rupture_Strain, Concrete_Modulus,
                              Confinement_Stiffness, Confinement_Stress, Strain_Ratio, Stiffness_Ratio]
            )
            
            # Additional outputs
            Strain_Enhancement = Confined_Strain / Unconfined_Strain if Unconfined_Strain != 0 else 0
            Strength_Enhancement = FRP_Strength_Prediction / Unconfined_Strength if Unconfined_Strength != 0 else 0
            
            st.success(f'Predicted Confined Strength: {FRP_Strength_Prediction}')
            st.info(f'Confined Strain: {Confined_Strain}')
            st.info(f'Strain Enhancement: {Strain_Enhancement}')
            st.info(f'Strength Enhancement: {Strength_Enhancement}')
            
        except ZeroDivisionError:
            st.error('Error: Division by zero. Please ensure non-zero values for input parameters.')

    # Informational tip
    st.info("Tip: Adjust input parameters and click the 'Predict Confined Strength' button.")

if __name__ == "__main__":
    main()
