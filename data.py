import numpy as np
import pandas as pd


def generate_dataset(disease_id, sample_size):
    np.random.seed(disease_id)
    multipliers = {
        'Age': 1,
        'BMI': 0.5,
        'Smoker': 7,
        'Drinker': 3,
        'Headache': 2,
        'Fatigue': 1.5,
        'Fever': 1,
        'ChestPain': 10,
        'ShortnessOfBreath': 8,
        'Dizziness': 2,
        'Palpitations': 6,
        'Swelling': 4,
        'IrregularHeartbeat': 7,
        'Nausea': 1,
        'ColdSweats': 3,
        'Indigestion': 1,
        'PainArm': 9,
        'JawPain': 5,
        'BackPain': 1.5,
        'Fainting': 7,
        'PersistentCough': 2,
        'DifficultySleeping': 4,
        'SuddenWeightGain': 3,
        'RapidPulse': 6,
        'ExtremeWeakness': 2,
        'LossOfConsciousness': 8,
        'BlurredVision': 2.5,
        'LegPain': 4,
        'NumbnessInLimbs': 3,
        'ColdLimbs': 3,
        'Cyanosis': 6,
        'PoorGrowthInInfants': 5,
        'RecurrentRespiratoryInfections': 4,
        'Sweating': 2,
        'Confusion': 2.5,
        'NumbnessOrWeaknessInLegs': 4,
        'ColdnessInLowerLegOrFoot': 3.5,
        'SoresOrWoundsOnToesFeetOrLegs': 5,
    }

    disease_symptoms = {
        1: ['ChestPain', 'ShortnessOfBreath', 'Fatigue', 'IrregularHeartbeat', 'ExtremeWeakness', 'BlurredVision'],  # CAD
        2: ['ChestPain', 'ShortnessOfBreath', 'Nausea', 'ColdSweats', 'PainArm', 'PoorGrowthInInfants'],  # Heart Attack
        3: ['ShortnessOfBreath', 'Fatigue', 'Swelling', 'PersistentCough', 'RapidPulse', 'JawPain'],  # Heart Failure
        4: ['Palpitations', 'Dizziness', 'Fainting', 'RapidPulse', 'Fatigue', 'ChestPain'],  # Arrhythmias
        5: ['Headache', 'Dizziness', 'BlurredVision', 'ShortnessOfBreath', 'Cyanosis', 'ColdSweats'],  # Hypertension
        6: ['ChestPain', 'LegPain', 'NumbnessInLimbs', 'ColdLimbs', 'ExtremeWeakness', 'Confusion'],  # Atherosclerosis
        7: ['ChestPain', 'ShortnessOfBreath', 'Fatigue', 'Nausea', 'IrregularHeartbeat', 'Cyanosis'],  # Angina Pectoris
        8: ['Fatigue', 'ShortnessOfBreath', 'ChestPain', 'Fever', 'BMI', 'ColdSweats'],  # Endocarditis
        9: ['ShortnessOfBreath', 'ChestPain', 'Fatigue', 'Dizziness', 'Swelling', 'Drinker'],  # Valvular Heart Disease
        10: ['ChestPain', 'Fever', 'ShortnessOfBreath', 'Fatigue', 'Drinker', 'RapidPulse'],  # Pericarditis
        11: ['ShortnessOfBreath', 'Fatigue', 'Swelling', 'IrregularHeartbeat', 'Dizziness', 'Indigestion'],  # Cardiomyopathy
        12: ['Cyanosis', 'ShortnessOfBreath', 'PoorGrowthInInfants', 'Fatigue', 'RecurrentRespiratoryInfections', 'Nausea'],  # Congenital Heart Defects
        13: ['ShortnessOfBreath', 'ChestPain', 'RapidPulse', 'Sweating', 'Indigestion', 'Fever'],  # Pulmonary Embolism
        14: ['LossOfConsciousness', 'Fainting', 'ExtremeWeakness', 'Nausea', 'BackPain', 'LegPain'],  # Cardiac Arrest
        15: ['RapidPulse', 'Fatigue', 'ColdSweats', 'Confusion', 'DifficultySleeping', 'RecurrentRespiratoryInfections'],  # Cardiogenic Shock
        16: ['Fatigue', 'ShortnessOfBreath', 'ChestPain', 'Fever', 'Sweating', 'IrregularHeartbeat'],  # Rheumatic Heart Disease
        17: ['Fatigue', 'ShortnessOfBreath', 'Palpitations', 'Swelling', 'ColdnessInLowerLegOrFoot', 'Cyanosis'],  # Heart Valve Regurgitation
        18: ['ChestPain', 'ShortnessOfBreath', 'Fatigue', 'Fainting', 'SoresOrWoundsOnToesFeetOrLegs', 'Nausea'],  # Heart Valve Stenosis
        19: ['ShortnessOfBreath', 'Fatigue', 'Dizziness', 'Swelling', 'Confusion', 'Drinker'],  # Heart Murmur
        20: ['LegPain', 'NumbnessOrWeaknessInLegs', 'ColdnessInLowerLegOrFoot', 'SoresOrWoundsOnToesFeetOrLegs', 'Age', 'Drinker']  # Peripheral Artery Disease (PAD)
    }
    for disease_id in disease_ids:
        np.random.seed(disease_id)
        data = {symptom: np.zeros(sample_size) for symptom in multipliers.keys()}

        for symptom in disease_symptoms[disease_id]:
            if symptom in multipliers:
                data[symptom] = np.random.rand(sample_size) * multipliers[symptom]

        for symptom in multipliers.keys():
            max_val = multipliers[symptom]
            if max_val != 0:
                data[symptom] /= max_val

        relevant_symptom_sums = np.zeros(sample_size)
        for symptom in disease_symptoms[disease_id]:
            relevant_symptom_sums += data[symptom]

        data['Diagnosis Result'] = np.zeros(sample_size)
        data['Diagnosis Result'][relevant_symptom_sums >= 2.0] = disease_id
        data['Diagnosis Result'] = data['Diagnosis Result'].astype(int)

        df = pd.DataFrame(data)
        df.to_csv('datasets/'f'{disease_id}.csv', index=False)


disease_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
generate_dataset(disease_ids, 1000)