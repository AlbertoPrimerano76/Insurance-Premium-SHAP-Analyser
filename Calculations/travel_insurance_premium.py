import pandas as pd
import numpy as np
from random import choice, randint, uniform
from tqdm.auto import tqdm
from datetime import datetime, timedelta

def calculate_base_premium(trip_cost, trip_duration):
    # Base premium is typically a percentage of the trip cost
    base_rate = 0.06  # 6% of trip cost
    duration_factor = min(trip_duration / 30, 1)  # Cap at 1 month
    return trip_cost * base_rate * duration_factor

def age_factor(age):
    if age < 30:
        return 1.0
    elif age < 40:
        return 1.1
    elif age < 50:
        return 1.2
    elif age < 60:
        return 1.3
    elif age < 70:
        return 1.5
    else:
        return 2.0

def destination_factor(destination):
    factors = {
        'Domestic': 1.0,
        'Canada/Mexico': 1.1,
        'Europe': 1.2,
        'Asia': 1.3,
        'South America': 1.3,
        'Africa': 1.4,
        'Australia/New Zealand': 1.3,
        'Antarctica': 1.5
    }
    return factors.get(destination, 1.2)  # Default to 1.2 if destination not found

def coverage_factor(coverage_type):
    factors = {
        'Basic': 1.0,
        'Standard': 1.2,
        'Comprehensive': 1.5
    }
    return factors.get(coverage_type, 1.2)  # Default to Standard if not found

def calculate_premium(trip_cost, trip_duration, age, destination, coverage_type, pre_existing_conditions=False):
    base_premium = calculate_base_premium(trip_cost, trip_duration)
    
    total_premium = base_premium * age_factor(age) * destination_factor(destination) * coverage_factor(coverage_type)
    
    if pre_existing_conditions:
        total_premium *= 1.5  # 50% increase for pre-existing conditions
    
    return round(total_premium, 2)

def generate_test_data(n):
    data = []
    destinations = ['Domestic', 'Canada/Mexico', 'Europe', 'Asia', 'South America', 'Africa', 'Australia/New Zealand', 'Antarctica']
    coverage_types = ['Basic', 'Standard', 'Comprehensive']
    
    for _ in tqdm(range(n), desc="Generating data", leave=False):
        trip_cost = randint(500, 10000)
        trip_duration = randint(1, 30)
        age = randint(18, 80)
        destination = choice(destinations)
        coverage_type = choice(coverage_types)
        pre_existing_conditions = choice([True, False])

        # Additional non-influencing features
        loyalty_program = choice([True, False])  # Whether the customer is part of a loyalty program
        travel_companion_count = randint(0, 5)   # Number of travel companions
        frequent_traveler = choice([True, False]) # If the person is a frequent traveler
        
        premium = calculate_premium(trip_cost, trip_duration, age, destination, coverage_type, pre_existing_conditions)
        
        data.append([trip_cost, trip_duration, age, destination, coverage_type, pre_existing_conditions, 
                     loyalty_program, travel_companion_count, frequent_traveler, premium])
    
    return pd.DataFrame(data, columns=['Trip Cost', 'Trip Duration', 'Age', 'Destination', 
                                       'Coverage Type', 'Pre-existing Conditions', 
                                       'Loyalty Program', 'Travel Companion Count', 'Frequent Traveler', 'Premium'])

# List of important features that impact the premium
important_features = {
    'Trip Cost', 'Trip Duration', 'Age', 'Destination', 'Coverage Type', 'Pre-existing Conditions'
}

# Non-important features that do not affect the premium
non_important_features = {
    'Loyalty Program', 'Travel Companion Count', 'Frequent Traveler'
}
