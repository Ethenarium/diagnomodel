import numpy as np
import pandas as pd

np.random.seed(0)

sample_size = 100

multipliers = {
    'age_multiplier': 1,
    'bmi_multiplier': 0.5,
    'smoker_multiplier': 7,
    'drinker_multiplier': 3,
    'headache_multiplier': 2,
    'chest_pain_or_discomfort_multiplier': 10,
    'shortness_of_breath_multiplier': 8,
    'fatigue_multiplier': 1.5,
    'fever_multiplier':1,
    'dizziness_or_lightheadedness_multiplier': 2,
    'palpitations_multiplier': 6,
    'swelling_in_legs_ankles_feet_multiplier': 4,
    'irregular_heartbeat_multiplier': 7,
    'nausea_or_vomiting_multiplier': 1,
    'cold_sweats_multiplier': 3,
    'indigestion_or_heartburn_multiplier': 1,
    'pain_spreads_to_arm_multiplier': 9,
    'jaw_pain_multiplier': 5,
    'back_pain_multiplier': 1.5,
    'fainting_multiplier': 7,
    'persistent_cough_multiplier': 2,
    'difficulty_sleeping_lying_flat_multiplier': 4,
    'sudden_weight_gain_multiplier': 3,
    'rapid_or_irregular_pulse_multiplier': 6,
    'extreme_weakness_multiplier': 2,
    'loss_of_consciousness_multiplier': 8,
    'blurred_vision_multiplier': 2.5,
    'leg_pain_multiplier': 4,
    'numbness_in_limbs_multiplier': 3,
    'cold_limbs_multiplier': 3,
    'cyanosis_multiplier': 6,
    'poor_growth_in_infants_multiplier': 5,
    'recurrent_respiratory_infections_multiplier': 4,
    'sweating_multiplier': 2,
    'confusion_multiplier': 2.5,
    'numbness_or_weakness_in_legs_multiplier': 4,
    'coldness_in_lower_leg_or_foot_multiplier': 3.5,
    'sores_or_wounds_on_toes_feet_or_legs_multiplier': 5,
}

disease_symptoms = {
    1: ['ChestPain', 'ShortnessOfBreath', 'Fatigue', 'IrregularHeartbeat', 'ExtremeWeakness'],  # CAD
    2: ['ChestPain', 'ShortnessOfBreath', 'Nausea', 'ColdSweats', 'PainArm'],  # Heart Attack
    3: ['ShortnessOfBreath', 'Fatigue', 'Swelling', 'PersistentCough', 'RapidPulse'],  # Heart Failure
    4: ['Palpitations', 'Dizziness', 'Fainting', 'RapidPulse', 'Fatigue'],  # Arrhythmias
    5: ['Headache', 'Dizziness', 'BlurredVision', 'ShortnessOfBreath'],  # Hypertension
    6: ['ChestPain', 'LegPain', 'NumbnessInLimbs', 'ColdLimbs'],  # Atherosclerosis
    7: ['ChestPain', 'ShortnessOfBreath', 'Fatigue', 'Nausea'],  # Angina Pectoris
    8: ['Fatigue', 'ShortnessOfBreath', 'ChestPain', 'Fever'],  # Endocarditis
    9: ['ShortnessOfBreath', 'ChestPain', 'Fatigue', 'Dizziness', 'Swelling'],  # Valvular Heart Disease
    10: ['ChestPain', 'Fever', 'ShortnessOfBreath', 'Fatigue'],  # Pericarditis
    11: ['ShortnessOfBreath', 'Fatigue', 'Swelling', 'IrregularHeartbeat', 'Dizziness'],  # Cardiomyopathy
    12: ['Cyanosis', 'ShortnessOfBreath', 'PoorGrowthInInfants', 'Fatigue', 'RecurrentRespiratoryInfections'],  # Congenital Heart Defects
    13: ['ShortnessOfBreath', 'ChestPain', 'RapidPulse', 'Sweating'],  # Pulmonary Embolism
    14: ['LossOfConsciousness', 'Fainting', 'ExtremeWeakness'],  # Cardiac Arrest
    15: ['RapidPulse', 'Fatigue', 'ColdSweats', 'Confusion'],  # Cardiogenic Shock
    16: ['Fatigue', 'ShortnessOfBreath', 'ChestPain', 'Fever'],  # Rheumatic Heart Disease
    17: ['Fatigue', 'ShortnessOfBreath', 'Palpitations', 'Swelling'],  # Heart Valve Regurgitation
    18: ['ChestPain', 'ShortnessOfBreath', 'Fatigue', 'Fainting'],  # Heart Valve Stenosis
    19: ['ShortnessOfBreath', 'Fatigue', 'Dizziness', 'Swelling'],  # Heart Murmur
    20: ['LegPain', 'NumbnessOrWeaknessInLegs', 'ColdnessInLowerLegOrFoot', 'SoresOrWoundsOnToesFeetOrLegs'],  # Peripheral Artery Disease (PAD)
}

diagnosis_results = np.full(sample_size, -1)  # Default to -1, meaning no diagnosis matches the criteria
age = np.random.randint(18, 80, sample_size)
bmi = np.random.uniform(15, 40, sample_size)
smoker = np.random.rand(sample_size)
drinker = np.random.rand(sample_size)
headache = np.random.rand(sample_size)
fatigue = np.random.rand(sample_size)
fever = np.random.rand(sample_size)
chest_pain = np.random.rand(sample_size)
shortness_of_breath = np.random.rand(sample_size)
dizziness = np.random.rand(sample_size)
palpitations = np.random.rand(sample_size)
swelling = np.random.rand(sample_size)
irregular_heartbeat = np.random.rand(sample_size)
nausea = np.random.rand(sample_size)
cold_sweats = np.random.rand(sample_size)
indigestion = np.random.rand(sample_size)
pain_arm = np.random.rand(sample_size)
jaw_pain = np.random.rand(sample_size)
back_pain = np.random.rand(sample_size)
fainting = np.random.rand(sample_size)
persistent_cough = np.random.rand(sample_size)
difficulty_sleeping = np.random.rand(sample_size)
sudden_weight_gain = np.random.rand(sample_size)
rapid_pulse = np.random.rand(sample_size)
extreme_weakness = np.random.rand(sample_size)
loss_of_consciousness = np.random.rand(sample_size)
blurred_vision = np.random.rand(sample_size)
leg_pain = np.random.rand(sample_size)
numbness_in_limbs = np.random.rand(sample_size)
cold_limbs = np.random.rand(sample_size)
cyanosis = np.random.rand(sample_size)
poor_growth_in_infants = np.random.rand(sample_size)
recurrent_respiratory_infections = np.random.rand(sample_size)
sweating = np.random.rand(sample_size)
confusion = np.random.rand(sample_size)
numbness_or_weakness_in_legs = np.random.rand(sample_size)
coldness_in_lower_leg_or_foot = np.random.rand(sample_size)
sores_or_wounds_on_toes_feet_or_legs = np.random.rand(sample_size)

age_m = age * multipliers['age_multiplier']
bmi_m = bmi * multipliers['bmi_multiplier']
smoker_m = smoker * multipliers['smoker_multiplier']
drinker_m = drinker * multipliers['drinker_multiplier']
headache_m = headache * multipliers['headache_multiplier']
fatigue_m = fatigue * multipliers['fatigue_multiplier']
fever_m = fever * multipliers['fever_multiplier']
chest_pain_m = chest_pain * multipliers['chest_pain_or_discomfort_multiplier']
shortness_of_breath_m = shortness_of_breath * multipliers['shortness_of_breath_multiplier']
dizziness_m = dizziness * multipliers['dizziness_or_lightheadedness_multiplier']
palpitations_m = palpitations * multipliers['palpitations_multiplier']
swelling_m = swelling * multipliers['swelling_in_legs_ankles_feet_multiplier']
irregular_heartbeat_m = irregular_heartbeat * multipliers['irregular_heartbeat_multiplier']
nausea_m = nausea * multipliers['nausea_or_vomiting_multiplier']
cold_sweats_m = cold_sweats * multipliers['cold_sweats_multiplier']
indigestion_m = indigestion * multipliers['indigestion_or_heartburn_multiplier']
pain_arm_m = pain_arm * multipliers['pain_spreads_to_arm_multiplier']
jaw_pain_m = jaw_pain * multipliers['jaw_pain_multiplier']
back_pain_m = back_pain * multipliers['back_pain_multiplier']
fainting_m = fainting * multipliers['fainting_multiplier']
persistent_cough_m = persistent_cough * multipliers['persistent_cough_multiplier']
difficulty_sleeping_m = difficulty_sleeping * multipliers['difficulty_sleeping_lying_flat_multiplier']
sudden_weight_gain_m = sudden_weight_gain * multipliers['sudden_weight_gain_multiplier']
rapid_pulse_m = rapid_pulse * multipliers['rapid_or_irregular_pulse_multiplier']
extreme_weakness_m = extreme_weakness * multipliers['extreme_weakness_multiplier']
loss_of_consciousness_m = loss_of_consciousness * multipliers['loss_of_consciousness_multiplier']
blurred_vision_m = blurred_vision * multipliers['blurred_vision_multiplier']
leg_pain_m = leg_pain * multipliers['leg_pain_multiplier']
numbness_in_limbs_m = numbness_in_limbs * multipliers['numbness_in_limbs_multiplier']
cold_limbs_m = cold_limbs * multipliers['cold_limbs_multiplier']
cyanosis_m = cyanosis * multipliers['cyanosis_multiplier']
poor_growth_in_infants_m = poor_growth_in_infants * multipliers['poor_growth_in_infants_multiplier']
recurrent_respiratory_infections_m = recurrent_respiratory_infections * multipliers['recurrent_respiratory_infections_multiplier']
sweating_m = sweating * multipliers['sweating_multiplier']
confusion_m = confusion * multipliers['confusion_multiplier']
numbness_or_weakness_in_legs_m = numbness_or_weakness_in_legs * multipliers['numbness_or_weakness_in_legs_multiplier']
coldness_in_lower_leg_or_foot_m = coldness_in_lower_leg_or_foot * multipliers['coldness_in_lower_leg_or_foot_multiplier']
sores_or_wounds_on_toes_feet_or_legs_m = sores_or_wounds_on_toes_feet_or_legs * multipliers['sores_or_wounds_on_toes_feet_or_legs_multiplier']

condition_sum = (age_m + bmi_m + smoker_m + drinker_m + headache_m + fatigue_m +
                 chest_pain_m + shortness_of_breath_m + dizziness_m + palpitations_m +
                 swelling_m + irregular_heartbeat_m + nausea_m + cold_sweats_m +
                 indigestion_m + pain_arm_m + jaw_pain_m + back_pain_m + fainting_m +
                 persistent_cough_m + difficulty_sleeping_m + sudden_weight_gain_m +
                 rapid_pulse_m + extreme_weakness_m + loss_of_consciousness_m +
                 blurred_vision_m + leg_pain_m + numbness_in_limbs_m + cold_limbs_m +
                 cyanosis_m + poor_growth_in_infants_m + recurrent_respiratory_infections_m +
                 sweating_m + confusion_m + numbness_or_weakness_in_legs_m +
                 coldness_in_lower_leg_or_foot_m + sores_or_wounds_on_toes_feet_or_legs_m)

condition_sum_normalized = (condition_sum / condition_sum.max()) * 100

for i in range(sample_size):
    if chest_pain_m[i] > 0.50 and shortness_of_breath_m[i] > 0.50 and fatigue_m[i] > 0.50 and irregular_heartbeat_m[i] > 0.50 and extreme_weakness_m[i] > 0.50:
        diagnosis_results[i] = 1
    elif chest_pain_m[i] > 0.50 and shortness_of_breath_m[i] > 0.50 and nausea_m[i] > 0.50 and cold_sweats_m[i] > 0.50 and pain_arm_m[i] > 0.50:
        diagnosis_results[i] = 2
    elif shortness_of_breath_m[i] > 0.50 and fatigue_m[i] > 0.50 and swelling_m[i] > 0.50 and persistent_cough_m[i] > 0.50 and rapid_pulse_m[i] > 0.50:
        diagnosis_results[i] = 3
    elif palpitations_m[i] > 0.50 and dizziness_m[i] > 0.50 and fainting_m[i] > 0.50 and rapid_pulse_m[i] > 0.50 and fatigue_m[i] > 0.50:
        diagnosis_results[i] = 4
    elif headache_m[i] > 0.50 and dizziness_m[i] > 0.50 and blurred_vision_m[i] > 0.50 and shortness_of_breath_m[i] > 0.50:
        diagnosis_results[i] = 5
    elif chest_pain_m[i] > 0.50 and leg_pain_m[i] > 0.50 and numbness_in_limbs_m[i] > 0.50 and cold_limbs_m[i] > 0.50:
        diagnosis_results[i] = 6
    elif chest_pain_m[i] > 0.50 and shortness_of_breath_m[i] > 0.50 and fatigue_m[i] > 0.50 and nausea_m[i] > 0.50:
        diagnosis_results[i] = 7
    elif fatigue_m[i] > 0.50 and shortness_of_breath_m[i] > 0.50 and chest_pain_m[i] > 0.50 and fever_m[i] > 0.50:
        diagnosis_results[i] = 8
    elif shortness_of_breath_m[i] > 0.50 and chest_pain_m[i] > 0.50 and fatigue_m[i] > 0.50 and dizziness_m[i] > 0.50 and swelling_m[i] > 0.50:
        diagnosis_results[i] = 9
    elif chest_pain_m[i] > 0.50 and fever_m[i] > 0.50 and shortness_of_breath_m[i] > 0.50 and fatigue_m[i] > 0.50:
        diagnosis_results[i] = 10
    elif shortness_of_breath_m[i] > 0.50 and fatigue_m[i] > 0.50 and swelling_m[i] > 0.50 and irregular_heartbeat_m[i] > 0.50 and dizziness_m[i] > 0.50:
        diagnosis_results[i] = 11
    elif cyanosis_m[i] > 0.50 and shortness_of_breath_m[i] > 0.50 and poor_growth_in_infants_m[i] > 0.50 and fatigue_m[i] > 0.50 and recurrent_respiratory_infections_m[i] > 0.50:
         diagnosis_results[i] = 12
    elif shortness_of_breath_m[i] > 0.50 and chest_pain_m[i] > 0.50 and rapid_pulse_m[i] > 0.50 and sweating_m[i] > 0.50:
        diagnosis_results[i] = 13
    elif loss_of_consciousness_m[i] > 0.50 and fainting_m[i] > 0.50 and extreme_weakness_m[i] > 0.50:
        diagnosis_results[i] = 14
    elif rapid_pulse_m[i] > 0.50 and fatigue_m[i] > 0.50 and cold_sweats_m[i] > 0.50 and confusion_m[i] > 0.50:
        diagnosis_results[i] = 15
    elif fatigue_m[i] > 0.50 and shortness_of_breath_m[i] > 0.50 and palpitations_m[i] > 0.50 and swelling_m[i] > 0.50:
        diagnosis_results[i] = 17
    elif chest_pain_m[i] > 0.50 and shortness_of_breath_m[i] > 0.50 and fatigue_m[i] > 0.50 and fainting_m[i] > 0.50:
        diagnosis_results[i] = 18
    elif shortness_of_breath_m[i] > 0.50 and fatigue_m[i] > 0.50 and dizziness_m[i] > 0.50 and swelling_m[i] > 0.50:
        diagnosis_results[i] = 19
    elif leg_pain_m[i] > 0.50 and numbness_or_weakness_in_legs_m[i] > 0.50 and coldness_in_lower_leg_or_foot_m[i] > 0.50 and sores_or_wounds_on_toes_feet_or_legs_m[i] > 0.50:
        diagnosis_results[i] = 20
    else:
        diagnosis_results[i] = 0

df = pd.DataFrame({
    'Age': age,
    'BMI': bmi,
    'Smoker': smoker,
    'Drinker': drinker,
    'Headache': headache,
    'Fatigue': fatigue,
    'ChestPain': chest_pain,
    'ShortnessOfBreath': shortness_of_breath,
    'Dizziness': dizziness,
    'Palpitations': palpitations,
    'Swelling': swelling,
    'IrregularHeartbeat': irregular_heartbeat,
    'Nausea': nausea,
    'ColdSweats': cold_sweats,
    'Indigestion': indigestion,
    'PainArm': pain_arm,
    'JawPain': jaw_pain,
    'BackPain': back_pain,
    'Fainting': fainting,
    'PersistentCough': persistent_cough,
    'DifficultySleeping': difficulty_sleeping,
    'SuddenWeightGain': sudden_weight_gain,
    'RapidPulse': rapid_pulse,
    'ExtremeWeakness': extreme_weakness,
    'LossOfConsciousness': loss_of_consciousness,
    'BlurredVision': blurred_vision,
    'LegPain': leg_pain,
    'NumbnessInLimbs': numbness_in_limbs,
    'ColdLimbs': cold_limbs,
    'Cyanosis': cyanosis,
    'PoorGrowthInInfants': poor_growth_in_infants,
    'RecurrentRespiratoryInfections': recurrent_respiratory_infections,
    'Sweating': sweating,
    'Confusion': confusion,
    'NumbnessOrWeaknessInLegs': numbness_or_weakness_in_legs,
    'ColdnessInLowerLegOrFoot': coldness_in_lower_leg_or_foot,
    'SoresOrWoundsOnToesFeetOrLegs': sores_or_wounds_on_toes_feet_or_legs,
    'Diagnosis Result': diagnosis_results
})

csv_file_path = 'patient_data.csv'
df.to_csv(csv_file_path, index=False)