import pandas as pd
import numpy as np
from random import choice, randint, uniform
from scipy.stats import norm, lognorm, weibull_min

# Define a complex travel insurance premium calculation function
def calculate_travel_insurance_premium(
    destination_risk, trip_duration, age, medical_conditions, travel_purpose, 
    claim_history, trip_cost, coverage_limit, loyalty_years, 
    region, accommodation_type, travel_season
):
    base_premium = 200  # Base premium for a standard trip
    
    # Region risk factor (with 80+ possible regions)
    region_risk_factors = {f"Region {i}": uniform(0.5, 2.0) for i in range(1, 81)}
    region_risk = region_risk_factors.get(region, 1.0)
    
    # Accommodation type risk factor (80+ possible accommodations)
    accommodation_risk_factors = {f"Accommodation {i}": uniform(0.7, 1.5) for i in range(1, 81)}
    accommodation_risk = accommodation_risk_factors.get(accommodation_type, 1.0)
    
    # Travel season risk factor (80+ possible seasons)
    season_risk_factors = {f"Season {i}": uniform(0.6, 1.8) for i in range(1, 81)}
    season_risk = season_risk_factors.get(travel_season, 1.0)

    # Risk adjustments based on medical conditions and claim history
    medical_conditions_factor = np.exp(0.1 * medical_conditions)  # Exponential relationship
    claim_history_factor = 1 + (0.05 * claim_history)  # Linear increase with claim history

    # Age factor (logarithmic impact)
    age_factor = np.log1p(age / 10) ** 2

    # Trip cost factor (log-normal distribution)
    trip_cost_factor = lognorm(s=0.5, scale=1).pdf(trip_cost / 1000)
    
    # Loyalty discount based on years with the company (non-linear)
    loyalty_discount = max(0.95 - (loyalty_years * 0.01), 0.7)  # Minimum 30% discount
    
    # Travel purpose adjustment (simplified for example)
    travel_purpose_factors = {'Leisure': 1.0, 'Business': 1.2, 'Adventure': 1.5, 'Family': 0.8}
    purpose_factor = travel_purpose_factors.get(travel_purpose, 1.0)

    # Combine all factors into the premium calculation
    adjusted_premium = (
        base_premium * region_risk * accommodation_risk * season_risk *
        medical_conditions_factor * claim_history_factor * age_factor *
        trip_cost_factor * purpose_factor
    )

    # Apply coverage limit adjustments (logarithmic)
    coverage_factor = np.log1p(coverage_limit / 5000)

    # Apply loyalty discount
    final_premium = adjusted_premium * coverage_factor * loyalty_discount

    return round(final_premium, 2)

# Generate test data for the travel insurance policy
def generate_travel_test_data(n):
    data = []
    regions = [f"Region {i}" for i in range(1, 81)]
    accommodations = [f"Accommodation {i}" for i in range(1, 81)]
    seasons = [f"Season {i}" for i in range(1, 81)]
    purposes = ['Leisure', 'Business', 'Adventure', 'Family']
    
    for _ in range(n):
        destination_risk = randint(1, 10)
        trip_duration = randint(1, 60)  # Days
        age = randint(18, 75)
        medical_conditions = randint(0, 10)  # Number of pre-existing conditions
        travel_purpose = choice(purposes)
        claim_history = randint(0, 5)
        trip_cost = uniform(500, 10000)
        coverage_limit = choice([10000, 20000, 50000, 100000])
        loyalty_years = randint(0, 20)
        region = choice(regions)
        accommodation_type = choice(accommodations)
        travel_season = choice(seasons)
        
        premium = calculate_travel_insurance_premium(
            destination_risk, trip_duration, age, medical_conditions, travel_purpose,
            claim_history, trip_cost, coverage_limit, loyalty_years,
            region, accommodation_type, travel_season
        )
        
        data.append([destination_risk, trip_duration, age, medical_conditions, travel_purpose, 
                     claim_history, trip_cost, coverage_limit, loyalty_years, region, 
                     accommodation_type, travel_season, premium])
    
    return pd.DataFrame(data, columns=['Destination Risk', 'Trip Duration', 'Age', 
                                       'Medical Conditions', 'Travel Purpose', 'Claim History', 
                                       'Trip Cost', 'Coverage Limit', 'Loyalty Years', 'Region', 
                                       'Accommodation Type', 'Travel Season', 'Premium'])

