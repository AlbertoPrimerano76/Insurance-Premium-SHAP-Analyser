import pandas as pd
import numpy as np
from random import choice, randint
from datetime import datetime, timedelta
from tqdm.auto import tqdm

# Dataset for pricing calculation
base_rate_data = {
    'Coverage': ['Liability', 'Windscreen', 'Any Repairer', 'No Claim Bonus Protection'],
    'Base Rate': [723, 20, 20, 30]
}

gender_factor_data = {
    'Gender': ['Male', 'Female'],
    'Monthly Cost': ['$103.80', '$99.80'],
    'Yearly Cost': ['$1,245.60', '$1,197.60']
}

state_territory_data = {
    'Territory': [
        'New South Wales', 'Queensland', 'Victoria', 'South Australia', 
        'Western Australia', 'ACT, Tasmania and NT*'
    ],
    'Monthly Cost': ['$120.00', '$85.70', '$98.60', '$103.20', '$88.60', '$93.63'],
    'Yearly Cost': ['$1,440.00', '$1,028.40', '$1,183.20', '$1,238.40', '$1,036.20', '$1,123.60'],
}

annual_km_data = {
    'Annual KM': [
        '0-5000 kms', '5001-10000 kms', '10001-15000 kms', '15001-20000 kms', '> 20000 kms'
    ],
    'Factor': [0.8, 1, 1.2, 1.4, 1.6]
}

business_use_data = {
    'Business Use': ['Private', 'Rideshare', 'Business'],
    'Factor': [1, 1.2, 1.4]
}

age_group_data = {
    'Age group': [
        '18-24 years old', '25-34 years old', '35-44 years old', '45-54 years old', 
        '55-64 years old', '65+ years old'
    ],
    'Monthly Cost': ['$142.70', '$114.20', '$105.20', '$106.20', '$91.20', '$72.70'],
    'Yearly Cost': ['$1,712.40', '$1,370.40', '$1,262.40', '$1,274.40', '$1,094.40', '$872.40'],
}

age_data = {
    'Age': list(range(18, 101)),
    'Age Group': [
        '18-24 years old']*7 \
            + ['25-34 years old']*10 \
                + ['35-44 years old']*10 \
                    + ['45-54 years old']*10 \
                        + ['55-64 years old']*10 \
                            + ['65+ years old']*36
}

def get_factor(lookup_df, parameter_column, factor_column, value):
    """
    Retrieve the factor value from the dataframe based on a specific parameter.
    
    :param lookup_df: DataFrame containing the lookup data
    :param parameter_column: Column name of the parameter
    :param factor_column: Column name of the factor
    :param value: Value to lookup
    :return: Factor value
    """
    if value not in lookup_df[parameter_column].values:
        raise ValueError(f"Value '{value}' not found in '{parameter_column}' column.")
    if factor_column not in lookup_df.columns:
        raise ValueError(f"Factor column '{factor_column}' not found in DataFrame.")
    factor = lookup_df.loc[lookup_df[parameter_column] == value, factor_column]
    if factor.empty:
        raise ValueError(f"No factor available for '{value}' in column '{factor_column}'.")
    return factor.values[0]

def calculate_factor(value):
    """
    Convert a monetary value string to a float and divide by 100.
    
    :param value: String representation of monetary value
    :return: Converted float value
    """
    return float(value.replace('$', '').replace(',', '')) / 100

# Define the input fields
input_fields = [
    'Annual kilometers', 'State', 'Business Use', 'Financing', 'Gender', 
    'Date of Birth', 'Year', 'Make Model', 'Parking Location', 'Name', 'Age',
    'Number of Claims', 'Payment Frequency', 'Hire Car', \
        'Sum Insured Value Type', 'Sum Insured Amount'
]

# Lookup dictionary with extracted values
lookup_dict = {
    'Private Passenger Coverage Type': ['Third Party Property Damage', \
        'Third Party Fire & Theft', 'Comprehensive'],
    'Sum Insured Value Type': ['Market Value', 'Agreed Value'],
    'Gender': ['Female', 'Male'],
    'Annual kilometers': ['0-5000 kms', '5001-10000 kms', \
        '10001-15000 kms', '15001-20000 kms', '> 20000 kms'],
    'Parking Location': ['Garage', 'Carport', 'Driveway', 'Street'],
    'Business Use': ['Private', 'Rideshare', 'Business'],
    'Payment Frequency': ['Annual', 'Monthly'],
    'Hire Car': ['Yes', 'No'], 
    'Windscreen': ['Yes', 'No'], 
    'Any Repairer': ['Yes', 'No'],
    'No Claim Bonus Protection': ['Yes', 'No']
}

def random_date(start, end):
    """
    Generate a random date between two dates.
    
    :param start: Start date
    :param end: End date
    :return: Random date between start and end
    """
    return start + timedelta(days=randint(0, int((end - start).days)))

def generate_random_data(n):
    """
    Generate random test data.
    
    :param n: Number of records to generate
    :return: DataFrame containing the generated data
    """
    data = []
    for _ in range(n):
        date = random_date(datetime(2022, 1, 1), datetime(2026, 12, 31))
        year = randint(2000, 2026)
        make_model = choice(['Toyota Corolla', 'Ford Focus', 'Honda Civic', 'Dodge Ram'])
        annual_kilometers = choice(lookup_dict['Annual kilometers'])
        state = choice(['New South Wales', 'Queensland', 'Victoria', 'South Australia', 
                        'Western Australia', 'ACT, Tasmania and NT*'])
        parking_location = choice(lookup_dict['Parking Location'])
        business_use = choice(lookup_dict['Business Use'])
        financing = choice(['Yes', 'No'])
        name = f'Name_{randint(1, 1000)}'
        gender = choice(lookup_dict['Gender'])
        dob = random_date(datetime(1940, 1, 1), date - timedelta(days=21*365))
        age = date.year - dob.year - ((date.month, date.day) < (dob.month, dob.day))
        number_of_claims = randint(0, 5)
        payment_frequency = choice(lookup_dict['Payment Frequency'])
        hire_car = choice(lookup_dict['Hire Car'])
        windscreen = choice(lookup_dict['Windscreen'])
        any_repairer = choice(lookup_dict['Any Repairer'])
        ncb_protection = choice(lookup_dict['No Claim Bonus Protection'])
        sum_insured_value_type = choice(lookup_dict['Sum Insured Value Type'])
        sum_insured_amount = randint(10000, 100000)
        data.append([
            annual_kilometers, state, business_use, financing, gender, 
            dob.strftime('%Y-%m-%d'), year, make_model, parking_location, name, 
            age, number_of_claims, payment_frequency, hire_car, 
            sum_insured_value_type, sum_insured_amount
        ])
    return pd.DataFrame(data, columns=input_fields)

def generate_test_data(n):
    """
   
    :param n: Number of records to generate
    :return: DataFrame containing the generated data with liability premiums
    """
    base_rate_df = pd.DataFrame(base_rate_data)
    gender_factor_df = pd.DataFrame(gender_factor_data)
    territory_factor_df = pd.DataFrame(state_territory_data)
    annual_km_df = pd.DataFrame(annual_km_data)
    business_use_df = pd.DataFrame(business_use_data)
    age_df = pd.DataFrame(age_data)
    age_group_df = pd.DataFrame(age_group_data)
    base_rate_factor = get_factor(base_rate_df, 'Coverage', 'Base Rate', 'Liability')
    liability_premiums = []
    test_data = generate_random_data(n)
    for _, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc="Processing rows", leave=False):
        gender = row['Gender']
        raw_gender_factor = get_factor(gender_factor_df, 'Gender', 'Monthly Cost', gender)
        gender_factor = calculate_factor(raw_gender_factor)
        territory = row['State']
        raw_territory_factor = \
            get_factor(territory_factor_df, 'Territory', 'Monthly Cost', territory)
        territory_factor = calculate_factor(raw_territory_factor)
        annual_km = row['Annual kilometers']
        annual_km_factor = get_factor(annual_km_df, 'Annual KM', 'Factor', annual_km)
        business_use = row['Business Use']
        business_use_factor = get_factor(business_use_df, 'Business Use', 'Factor', business_use)
        birth_date = datetime.strptime(row['Date of Birth'], '%Y-%m-%d')
        today = datetime.today()
        age = today.year - birth_date.year \
            - ((today.month, today.day) < (birth_date.month, birth_date.day))
        age_group = get_factor(age_df, 'Age', 'Age Group', age)
        raw_age_group_factor = get_factor(age_group_df, 'Age group', 'Monthly Cost', age_group)
        age_group_factor = calculate_factor(raw_age_group_factor)
        liability_premium = round(
            base_rate_factor * gender_factor \
                * age_group_factor * territory_factor \
                    * annual_km_factor * business_use_factor, 0
        )
        liability_premiums.append(liability_premium)
        tqdm._instances.clear()
    test_data['Premium'] = liability_premiums
    return test_data
