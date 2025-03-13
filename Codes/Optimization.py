import joblib
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import math
from Visualization_class import * # Defined calss for visualization

df_cities = pd.read_csv('data/simulation_data.csv')
df_cities.drop('city',axis = 1, inplace = True)
df_cities_ = df_cities.iloc[:,3:]


# === Load the trained ExtraTrees model ===
model_path = "models/et_model_ET.pkl"  # Update path if needed
loaded_model = joblib.load(model_path)

# ===  Define multiple fixed sensor locations ===
sensors = {
    1: {'SP-Soth-Dis': 3.8, 'SP-East-Dis': 3, 'SP-North-Dis': 4.2, 'SP-West-Dis': 7},
#    2: {'SP-Soth-Dis': 4, 'SP-East-Dis': 5, 'SP-North-Dis': 4, 'SP-West-Dis': 5}, # Can be added to number of sensors
    3: {'SP-Soth-Dis': 4.2, 'SP-East-Dis': 7, 'SP-North-Dis': 4.2, 'SP-West-Dis': 3},
}

# === Climate & Environmental Conditions (Same for all sensors) ===

l = 4 #Index of simulation

environmental_features = {}


for i in df_cities_.columns:
    environmental_features[i] = df_cities_.loc[l,i]

# === Optimization Targets: A SINGLE Attractor Point ===
AP_LENGTH_BOUNDS = (0, 10)
AP_WIDTH_BOUNDS = (0, 3)

# === Boundary Check Function ===
def check_bounds(individual):
    """ Ensures AP-Length and AP-Width remain within valid bounds """
    individual[0] = max(AP_LENGTH_BOUNDS[0], min(individual[0], AP_LENGTH_BOUNDS[1]))  # AP-Length
    individual[1] = max(AP_WIDTH_BOUNDS[0], min(individual[1], AP_WIDTH_BOUNDS[1]))  # AP-Width
    return individual  

# === Objective Function Enforcing Et Constraints ===
def objective_function(individual):
    ap_length, ap_width = individual  # Optimization variables (Single Attractor Point)

    total_et = 0  # Sum of `Et` values for all sensors
    penalty = -1e6  # Strong penalty for bad solutions

    for sensor_id, sensor in sensors.items():
        # Compute Distance and Angle
        sp_ap_dis = math.sqrt((ap_length - sensor['SP-East-Dis'])**2 + (ap_width - sensor['SP-Soth-Dis'])**2)
        angle = math.degrees(math.atan2((ap_width - sensor['SP-Soth-Dis']), (ap_length - sensor['SP-East-Dis'])))

        # Construct feature vector
        feature_vector = {
            **sensor, 'SP-Ap-Dis': sp_ap_dis, 'Angle': angle,
            'AP-Length': ap_length, 'AP-Width': ap_width,
            **environmental_features
        }

        # Predict Et
        feature_df = pd.DataFrame([feature_vector])
        predicted_et = loaded_model.predict(feature_df)[0]

        # If Et is outside [500, 1000], assign a strong penalty
        if predicted_et < 500 or predicted_et > 1000:
            return penalty,  # Immediately discard this solution

        # Sum up Et values
        total_et += predicted_et  

    return -total_et,  # Negative for minimization (since NSGA-II minimizes)

# === Set up NSGA-II Optimization ===
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize Et
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, *AP_LENGTH_BOUNDS)  
toolbox.register("attr_float2", np.random.uniform, *AP_WIDTH_BOUNDS)
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_float, toolbox.attr_float2), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# === Register Mutation and Crossover with Bound Checking ===
def mutate_with_bounds(individual):
    mutated_ind = tools.mutGaussian(individual, mu=0, sigma=0.5, indpb=0.2)  
    return check_bounds(mutated_ind[0]),  

def mate_with_bounds(ind1, ind2):
    offspring = tools.cxBlend(ind1, ind2, alpha=0.5)  
    return check_bounds(offspring[0]), check_bounds(offspring[1])  

toolbox.register("mutate", mutate_with_bounds)
toolbox.register("mate", mate_with_bounds)

toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", objective_function)

# === 🟢 NSGA-II Hyperparameters ===
POP_SIZE = 50  
NGEN = 100  
CX_PROB = 0.7  
MUT_PROB = 0.2  

# Initialize population
population = toolbox.population(n=POP_SIZE)

# Run the NSGA-II optimization
algorithms.eaMuPlusLambda(population, toolbox, mu=POP_SIZE, lambda_=POP_SIZE, 
                          cxpb=CX_PROB, mutpb=MUT_PROB, ngen=NGEN, 
                          stats=None, halloffame=None, verbose=True)

# Extract the best solution
best_solution = tools.selBest(population, k=1)[0]
optimal_ap_length, optimal_ap_width = best_solution

# ===Evaluate Sensors with Optimized AP Location and Print Et Values ===
print("\n=== Optimized Sensor Et Values ===")
for sensor_id, sensor in sensors.items():
    sp_ap_dis = math.sqrt((optimal_ap_length - sensor['SP-East-Dis'])**2 + (optimal_ap_width - sensor['SP-Soth-Dis'])**2)
    angle = math.degrees(math.atan2((optimal_ap_width - sensor['SP-Soth-Dis']), (optimal_ap_length - sensor['SP-East-Dis'])))

    feature_vector = {
        **sensor, 'SP-Ap-Dis': sp_ap_dis, 'Angle': angle,
        'AP-Length': optimal_ap_length, 'AP-Width': optimal_ap_width,
        **environmental_features
    }

    feature_df = pd.DataFrame([feature_vector])
    predicted_et = loaded_model.predict(feature_df)[0]

    # Print Et value for each sensor
    print(f"Sensor {sensor_id}: Et = {predicted_et:.2f}")

    # Print warning if Et is outside range
    if predicted_et < 500:
        print(f"For Sensor {sensor_id}, daylight is **not sufficient** (Et = {predicted_et:.2f})")
    elif predicted_et > 1000:
        print(f"For Sensor {sensor_id}, daylight is **excessive** (Et = {predicted_et:.2f})")

print(f"\n=== Optimal Attractor Point Location ===")
print(f"Optimal AP-Length: {optimal_ap_length:.4f}")
print(f"Optimal AP-Width: {optimal_ap_width:.4f}")

df = pd.read_csv('data/simulation_data.csv')
df = df.query('seed == 2') # seed can be any number

# Create DataFrame for given sensor locations
sensor_data = df.iloc[:,1:5]

sensor_data['AP-Length'] = optimal_ap_length
sensor_data['AP-Width'] = optimal_ap_width

# Calculate SP-Ap-Dis and Angle for each sensor
sensor_data['SP-Ap-Dis'] = sensor_data.apply(lambda row: math.sqrt(
    (optimal_ap_length - row['SP-East-Dis'])**2 + (optimal_ap_width - row['SP-Soth-Dis'])**2
), axis=1)

sensor_data['Angle'] = sensor_data.apply(lambda row: math.degrees(math.atan2(
    (optimal_ap_width - row['SP-Soth-Dis']), (optimal_ap_length - row['SP-East-Dis'])
)), axis=1)

sensor_data.reset_index(drop = True, inplace=True)

df_climate = pd.concat([pd.DataFrame(df_cities_.iloc[l,:]).T] * 500, ignore_index=True)

data = pd.concat([df_climate, sensor_data], ignore_index=False, axis=1)

# Order of columns acceptable for trained model
desired_order = [
    'SP-Soth-Dis', 'SP-East-Dis', 'SP-North-Dis', 'SP-West-Dis', 
    'SP-Ap-Dis', 'Angle', 'AP-Length', 'AP-Width', 
    'Dir-wea', 'Diff-wea', 'glob-wea', 'Total-Sky-cover', 
    'Altitude', 'Azimuth', 'dry_bulb_temperature', 'relative_humidity', 
    'wind_speed', 'latitude', 'longitude', 'elevation'
]

# Reorder the columns based on the desired order (excluding 'city' as it was not in the feature vector)
data = data[desired_order]

# Predict value for data collected after optimization
heatmap_values = loaded_model.predict(data).reshape(25, 20)

# Creat an instance of the SensorGridVisualization class
viz = SensorGridVisualization()

viz.visualize(heatmap_values,[0,data['AP-Length'][0],data['AP-Width'][0]])

