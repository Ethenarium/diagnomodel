from tensorflow.keras.models import load_model
import numpy as np

cad_model = load_model('cad_model.keras')
heart_attack_model = load_model('heart_attack_model.keras')
heart_failure_model = load_model('heart_failure_model.keras')
arrhythmias_model = load_model('arrhythmias_model.keras')
hypertension_model = load_model('hypertension_model.keras')
atherosclerosis_model = load_model('atherosclerosis_model.keras')
angina_pectoris_model = load_model('angina_pectoris_model.keras')
endocarditis_model = load_model('endocarditis_model.keras')
valvular_heart_disease_model = load_model('valvular_heart_disease_model.keras')
pericarditis_model = load_model('pericarditis_model.keras')

def classify_symptoms(symptoms):
    diseases = {
        'Cad': ['ChestPain', 'ShortnessOfBreath', 'Fatigue', 'IrregularHeartbeat', 'ExtremeWeakness', 'BlurredVision'], # CAD
        'Heart Attack': ['ChestPain', 'ShortnessOfBreath', 'Nausea', 'ColdSweats', 'PainArm', 'PoorGrowthInInfants'], # Heart Attack
        'Heart Failure': ['ShortnessOfBreath', 'Fatigue', 'Swelling', 'PersistentCough', 'RapidPulse', 'JawPain'], # Heart Failure
        'Arrhythmias': ['Palpitations', 'Dizziness', 'Fainting', 'RapidPulse', 'Fatigue', 'ChestPain'],  # Arrhythmias
        'Hypertension': ['Headache', 'Dizziness', 'BlurredVision', 'ShortnessOfBreath', 'Cyanosis', 'ColdSweats'],  # Hypertension
        'Atherosclerosis': ['ChestPain', 'LegPain', 'NumbnessInLimbs', 'ColdLimbs', 'ExtremeWeakness', 'Confusion'],  # Atherosclerosis
        'Angina Pectoris': ['ChestPain', 'ShortnessOfBreath', 'Fatigue', 'Nausea', 'IrregularHeartbeat', 'Cyanosis'],  # Angina Pectoris
        'Endocarditis': ['Fatigue', 'ShortnessOfBreath', 'ChestPain', 'Fever', 'BMI', 'ColdSweats'],  # Endocarditis
        'Valvular Heart Disease': ['ShortnessOfBreath', 'ChestPain', 'Fatigue', 'Dizziness', 'Swelling', 'Drinker'],  # Valvular Heart Disease
        'Pericarditis': ['ChestPain', 'Fever', 'ShortnessOfBreath', 'Fatigue', 'Drinker', 'RapidPulse'],  # Pericarditis
        'Cardiomyopathy': ['ShortnessOfBreath', 'Fatigue', 'Swelling', 'IrregularHeartbeat', 'Dizziness','Indigestion'],  # Cardiomyopathy
        'Congenital Heart Defects': ['Cyanosis', 'ShortnessOfBreath', 'PoorGrowthInInfants', 'Fatigue','RecurrentRespiratoryInfections', 'Nausea'],  # Congenital Heart Defects
        'Pulmonary Embolism': ['ShortnessOfBreath', 'ChestPain', 'RapidPulse', 'Sweating', 'Indigestion', 'Fever'],  # Pulmonary Embolism
        'Cardiac Arrest': ['LossOfConsciousness', 'Fainting', 'ExtremeWeakness', 'Nausea', 'BackPain', 'LegPain'],  # Cardiac Arrest
        'Cardiogenic Shock': ['RapidPulse', 'Fatigue', 'ColdSweats', 'Confusion', 'DifficultySleeping','RecurrentRespiratoryInfections'],  # Cardiogenic Shock
        'Rheumatic Heart Disease': ['Fatigue', 'ShortnessOfBreath', 'ChestPain', 'Fever', 'Sweating','IrregularHeartbeat'],  # Rheumatic Heart Disease
        'Heart Valve Regurgitation': ['Fatigue', 'ShortnessOfBreath', 'Palpitations', 'Swelling', 'ColdnessInLowerLegOrFoot', 'Cyanosis'],  # Heart Valve Regurgitation
        'Heart Valve Stenosis': ['ChestPain', 'ShortnessOfBreath', 'Fatigue', 'Fainting', 'SoresOrWoundsOnToesFeetOrLegs', 'Nausea'],  # Heart Valve Stenosis
        'Heart Murmur': ['ShortnessOfBreath', 'Fatigue', 'Dizziness', 'Swelling', 'Confusion', 'Drinker'], # Heart Murmur
        'Peripheral Artery Disease (PAD)': ['LegPain', 'NumbnessOrWeaknessInLegs', 'ColdnessInLowerLegOrFoot', 'SoresOrWoundsOnToesFeetOrLegs', 'Age', 'Drinker'] # Peripheral Artery Disease (PAD)
    }

    best_match = None
    max_count = 0
    max_intensity = 0

    for disease, relevant_symptoms in diseases.items():
        symptom_count = sum(symptom in symptoms for symptom in relevant_symptoms)
        symptom_intensity = sum(symptoms[symptom] for symptom in relevant_symptoms if symptom in symptoms)

        if symptom_count > max_count or (symptom_count == max_count and symptom_intensity > max_intensity):
            best_match = disease
            max_count = symptom_count
            max_intensity = symptom_intensity

    return best_match


def process_input(symptoms):
    disease = classify_symptoms(symptoms)
    if disease:
        model = load_model(f"{disease.lower().replace(' ', '_')}_model.keras")
        input_data = np.array([list(symptoms.values())])
        prediction = model.predict(input_data)
        return disease, prediction[0]
    return 'No disease detected or insufficient data', None

symptoms = {
    'Age': 0,
    'BMI': 0,
    'Smoker': 0,
    'Drinker': 0,
    'Headache': 0,
    'Fatigue': 0,
    'Fever': 0,
    'ChestPain': 0.66,
    'ShortnessOfBreath': 0.66,
    'Dizziness': 0,
    'Palpitations': 0,
    'Swelling': 0,
    'IrregularHeartbeat': 0,
    'Nausea': 0.66,
    'ColdSweats': 0.66,
    'Indigestion': 0,
    'PainArm': 0.66,
    'JawPain': 0,
    'BackPain': 0,
    'Fainting': 0,
    'PersistentCough': 0,
    'DifficultySleeping': 0,
    'SuddenWeightGain': 0,
    'RapidPulse': 0,
    'ExtremeWeakness': 0,
    'LossOfConsciousness': 0,
    'BlurredVision': 0,
    'LegPain': 0,
    'NumbnessInLimbs': 0,
    'ColdLimbs': 0,
    'Cyanosis': 0,
    'PoorGrowthInInfants': 0.66,
    'RecurrentRespiratoryInfections': 0,
    'Sweating': 0,
    'Confusion': 0,
    'NumbnessOrWeaknessInLegs': 0,
    'ColdnessInLowerLegOrFoot': 0,
    'SoresOrWoundsOnToesFeetOrLegs': 0,
}


disease, prediction = process_input(symptoms)
print(f"Disease predicted: {disease}, Probability: {prediction}")