import pandas as pd
import numpy as np
from scipy.stats import poisson, gamma
from random import choice, randint, uniform
from datetime import datetime, timedelta
from tqdm.auto import tqdm


def calculate_breach_probability(company_size, industry_risk, security_score):
    # Use a logistic function to calculate breach probability
    x = 0.1 * company_size + 2 * industry_risk - 0.5 * security_score
    return 1 / (1 + np.exp(-x))

def calculate_expected_loss(data_sensitivity, business_interruption_cost):
    # Use a combination of Poisson and Gamma distributions
    expected_breaches = poisson.rvs(mu=2)  # Expect 2 breaches per year on average
    loss_per_breach = gamma.rvs(a=2, scale=data_sensitivity * business_interruption_cost / 1000)
    return expected_breaches * loss_per_breach

def calculate_premium(company_size, industry_risk, security_score, data_sensitivity, business_interruption_cost):
    base_premium = 10000  # Base premium of $10,000
    
    breach_probability = calculate_breach_probability(company_size, industry_risk, security_score)
    expected_loss = calculate_expected_loss(data_sensitivity, business_interruption_cost)
    
    # Calculate premium using a combination of probability and expected loss
    risk_adjusted_premium = base_premium * (1 + np.log1p(breach_probability * expected_loss / 10000))
    
    # Apply a market competition factor (assuminnp.exp(-company_size / 1000)g more competition for larger companies)
    market_factor = np.exp(-company_size / 1000)
    
    final_premium = risk_adjusted_premium * market_factor
    
    return round(final_premium, 2)

def generate_test_data(n):
    data = []
    
    geographic_locations = ['North America', 'Europe', 'Asia', 'South America', 'Africa', 'Australia']
    types_of_data_stored = ['Personal', 'Financial', 'Health', 'Intellectual Property', 'Operational']
    compliance_standards = ['ISO 27001', 'NIST', 'GDPR', 'HIPAA', 'None']
    
    for _ in tqdm(range(n), desc="Generating data", leave=False):
        company_size = randint(10, 10000)  # Number of employees
        industry_risk = uniform(0.5, 5)  # Industry risk score from 0.5 to 5
        security_score = randint(1, 100)  # Security posture score from 1 to 100
        data_sensitivity = randint(1, 10)  # Data sensitivity score from 1 to 10
        business_interruption_cost = randint(1000, 1000000)  # Daily cost of business interruption
        
        # Additional factors not used in premium calculation
        geographic_location = choice(geographic_locations)
        previous_cyber_incidents = randint(0, 10)  # Number of previous cyber incidents
        types_of_data = choice(types_of_data_stored)
        compliance = choice(compliance_standards)
        
        premium = calculate_premium(company_size, industry_risk, security_score, data_sensitivity, business_interruption_cost)
        
        data.append([company_size, industry_risk, security_score, data_sensitivity, business_interruption_cost, 
                     geographic_location, previous_cyber_incidents, types_of_data, compliance, premium])
    
    return pd.DataFrame(data, columns=['Company Size', 'Industry Risk', 'Security Score', 'Data Sensitivity', 
                                       'Business Interruption Cost', 'Geographic Location', 
                                       'Previous Cyber Incidents', 'Types of Data Stored', 
                                       'Compliance Standard', 'Premium'])
