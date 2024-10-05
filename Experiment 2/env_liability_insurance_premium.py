import pandas as pd
import numpy as np
from scipy.stats import weibull_min
from skfuzzy import control as ctrl
from skfuzzy import membership as mf
from random import choice, randint, uniform
from tqdm.auto import tqdm

from datetime import datetime, timedelta

def create_fuzzy_system():
    # Input variables
    pollution_risk = ctrl.Antecedent(np.arange(0, 11, 1), 'pollution_risk')
    regulatory_compliance = ctrl.Antecedent(np.arange(0, 101, 1), 'regulatory_compliance')
    
    # Output variable
    risk_factor = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'risk_factor')
    
    # Membership functions
    pollution_risk['low'] = mf.trimf(pollution_risk.universe, [0, 0, 5])
    pollution_risk['medium'] = mf.trimf(pollution_risk.universe, [0, 5, 10])
    pollution_risk['high'] = mf.trimf(pollution_risk.universe, [5, 10, 10])

    regulatory_compliance['poor'] = mf.trimf(regulatory_compliance.universe, [0, 0, 50])
    regulatory_compliance['average'] = mf.trimf(regulatory_compliance.universe, [0, 50, 100])
    regulatory_compliance['good'] = mf.trimf(regulatory_compliance.universe, [50, 100, 100])

    risk_factor['low'] = mf.trimf(risk_factor.universe, [0, 0, 0.5])
    risk_factor['medium'] = mf.trimf(risk_factor.universe, [0, 0.5, 1])
    risk_factor['high'] = mf.trimf(risk_factor.universe, [0.5, 1, 1])

    # Fuzzy rules
    rule1 = ctrl.Rule(pollution_risk['low'] & regulatory_compliance['good'], risk_factor['low'])
    rule2 = ctrl.Rule(pollution_risk['high'] | regulatory_compliance['poor'], risk_factor['high'])
    rule3 = ctrl.Rule(pollution_risk['medium'], risk_factor['medium'])

    # Control system
    risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    return ctrl.ControlSystemSimulation(risk_ctrl)

fuzzy_system = create_fuzzy_system()

def environmental_impact_factor(years_of_operation, incident_history):
    # Weibull distribution to model increasing risk over time
    shape, scale = 1.5, 10
    time_factor = weibull_min.cdf(years_of_operation, shape, scale=scale)
    
    # Exponential growth factor based on incident history
    incident_factor = np.exp(0.1 * incident_history) - 1
    
    return time_factor * (1 + incident_factor)

def calculate_premium(industry_type, company_size, pollution_risk, regulatory_compliance, 
                      years_of_operation, incident_history, coverage_limit):
    base_premium = 50000  # Base premium of $50,000

    # Industry risk matrix (simplified)
    industry_risk = {
        'Manufacturing': 1.5,
        'Chemical': 2.0,
        'Agriculture': 1.2,
        'Energy': 1.8,
        'Transportation': 1.3
    }

    # Company size factor
    size_factor = np.log1p(company_size) / 10

    # Use fuzzy logic for risk assessment
    fuzzy_system.input['pollution_risk'] = pollution_risk
    fuzzy_system.input['regulatory_compliance'] = regulatory_compliance
    fuzzy_system.compute()
    risk_factor = fuzzy_system.output['risk_factor']

    # Calculate environmental impact
    impact_factor = environmental_impact_factor(years_of_operation, incident_history)

    # Combine all factors
    adjusted_premium = base_premium * industry_risk[industry_type] * size_factor * risk_factor * impact_factor

    # Adjust for coverage limit (assuming a log relationship)
    coverage_factor = np.log1p(coverage_limit / 1000000)  # Normalize to millions
    
    final_premium = adjusted_premium * coverage_factor

    return round(final_premium, 2)

def generate_test_data(n):
    data = []
    industry_types = ['Manufacturing', 'Chemical', 'Agriculture', 'Energy', 'Transportation']
    geographic_locations = ['Urban', 'Rural', 'Suburban', 'Coastal', 'Mountain']
    pollutants_handled = ['Heavy Metals', 'Organic Compounds', 'Pesticides', 'Fertilizers', 'Solvents']
    proximity_to_sensitive_areas = ['Close', 'Moderate', 'Far']
    climate_risk_factors = ['Low', 'Moderate', 'High']
    
    for _ in tqdm(range(n), desc="Generating data", leave=False):
        industry_type = choice(industry_types)
        company_size = randint(10, 10000)  # Number of employees
        pollution_risk = uniform(0, 10)  # Pollution risk score from 0 to 10
        regulatory_compliance = randint(0, 100)  # Compliance score from 0 to 100
        years_of_operation = randint(1, 100)
        incident_history = randint(0, 10)  # Number of past incidents
        coverage_limit = choice([1000000, 5000000, 10000000, 50000000, 100000000])  # Coverage limit in dollars
        
        # Additional factors
        geographic_location = choice(geographic_locations)
        pollutant_handled = choice(pollutants_handled)
        proximity_area = choice(proximity_to_sensitive_areas)
        climate_risk = choice(climate_risk_factors)
        
        premium = calculate_premium(industry_type, company_size, pollution_risk, regulatory_compliance, 
                                    years_of_operation, incident_history, coverage_limit)
        
        data.append([industry_type, company_size, pollution_risk, regulatory_compliance, 
                     years_of_operation, incident_history, coverage_limit, premium,
                     geographic_location, pollutant_handled, proximity_area, climate_risk])
    
    return pd.DataFrame(data, columns=['Industry Type', 'Company Size', 'Pollution Risk', 
                                       'Regulatory Compliance', 'Years of Operation', 
                                       'Incident History', 'Coverage Limit', 'Premium',
                                       'Geographic Location', 'Pollutants Handled', 
                                       'Proximity to Sensitive Areas', 'Climate Change Risk'])
