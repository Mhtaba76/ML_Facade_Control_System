import joblib
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import math

# === 🟢 Load Data ===
l = 6

df_cities1 = pd.read_csv('/Users/mrajaian/Downloads/cities_data.csv')
df_cities = df_cities1.drop('city', axis=1)
df_cities_ = df_cities.iloc[l, 3:]
city = df_cities1.iloc[l, 9]
hour = df_cities1.iloc[l, 2]
day = df_cities1.iloc[l, 1]
month = df_cities1.iloc[l, 0]
# === 🟢 Load the trained ExtraTrees models (Et and Ev) ===
et_model_path = "extra_trees_model.pkl"  
ev_model_path = "/Users/mrajaian/Downloads/et_model_Ev.pkl"

loaded_et_model = joblib.load(et_model_path)
loaded_ev_model = joblib.load(ev_model_path)

# === 🟢 Define fixed sensor locations ===
sensors = {
    1: {'SP-Soth-Dis': 3.8, 'SP-East-Dis': 3, 'SP-North-Dis': 4.2, 'SP-West-Dis': 7},
    3: {'SP-Soth-Dis': 4.2, 'SP-East-Dis': 7, 'SP-North-Dis': 4.2, 'SP-West-Dis': 3},
}

# === 🟢 Optimization Bounds ===
AP_LENGTH_BOUNDS = (0, 10)
AP_WIDTH_BOUNDS = (0, 3)

# === 🟢 Boundary Check Function ===
def check_bounds(individual):
    individual[0] = max(AP_LENGTH_BOUNDS[0], min(individual[0], AP_LENGTH_BOUNDS[1]))  # AP-Length
    individual[1] = max(AP_WIDTH_BOUNDS[0], min(individual[1], AP_WIDTH_BOUNDS[1]))  # AP-Width
    return individual  

# === 🟢 Multi-Objective Fitness (Maximize Et, Minimize Ev) ===
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # Maximize Et, Minimize Ev
creator.create("Individual", list, fitness=creator.FitnessMulti)

# === 🟢 Define Objective Function ===
def objective_function(individual):
    ap_length, ap_width = individual  

    valid_sensors = 0
    total_et = 0
    total_ev = 0
    penalty = -1e12  

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

        feature_df = pd.DataFrame([feature_vector])
        predicted_et = loaded_et_model.predict(feature_df)[0]
        predicted_ev = loaded_ev_model.predict(feature_df)[0]

        # Reject solutions where Ev is too high
        if predicted_ev >= 1000:
            return (penalty, penalty)  

        total_et += predicted_et
        total_ev += predicted_ev
        valid_sensors += 1  

    if valid_sensors > 0:
        avg_et = total_et / valid_sensors
        avg_ev = total_ev / valid_sensors
        return (avg_et, -avg_ev)  # Maximize Et, Minimize Ev
    else:
        return (penalty, penalty)  # Discard invalid solutions

# === 🟢 Set up DEAP Optimization ===
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, *AP_LENGTH_BOUNDS)  
toolbox.register("attr_float2", np.random.uniform, *AP_WIDTH_BOUNDS)
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_float, toolbox.attr_float2), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# === 🟢 Register Mutation and Crossover with Bound Checking ===
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
NGEN = 50  
CX_PROB = 0.7  
MUT_PROB = 0.2  

# === 🟢 Run Optimization ===
population = toolbox.population(n=POP_SIZE)
algorithms.eaMuPlusLambda(population, toolbox, mu=POP_SIZE, lambda_=POP_SIZE, 
                          cxpb=CX_PROB, mutpb=MUT_PROB, ngen=NGEN, 
                          stats=None, halloffame=None, verbose=True)

# === 🟢 Extract Pareto Front Solutions ===
pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

# === 🟢 Print Pareto Front ===
print("\n=== Pareto Front Solutions ===")
for ind in pareto_front:
    print(f"AP-Length: {ind[0]:.4f}, AP-Width: {ind[1]:.4f}, Et: {ind.fitness.values[0]:.2f}, Ev: {-ind.fitness.values[1]:.2f}")

# === 🟢 Evaluate Sensors with the Best Pareto Front Solutions ===
print("\n=== Evaluating Pareto Front Solutions ===")
for ind in pareto_front:
    ap_length, ap_width = ind  
    print(f"\nSolution -> AP-Length: {ap_length:.4f}, AP-Width: {ap_width:.4f}")

    for sensor_id, sensor in sensors.items():
        sp_ap_dis = math.sqrt((ap_length - sensor['SP-East-Dis'])**2 + (ap_width - sensor['SP-Soth-Dis'])**2)
        angle = math.degrees(math.atan2((ap_width - sensor['SP-Soth-Dis']), (ap_length - sensor['SP-East-Dis'])))

        feature_vector = {
            **sensor, 'SP-Ap-Dis': sp_ap_dis, 'Angle': angle,
            'AP-Length': ap_length, 'AP-Width': ap_width,
            **environmental_features
        }

        feature_df = pd.DataFrame([feature_vector])
        predicted_et = loaded_et_model.predict(feature_df)[0]
        predicted_ev = loaded_ev_model.predict(feature_df)[0]

        print(f"Sensor {sensor_id}: Et = {predicted_et:.2f}, Ev = {predicted_ev:.2f}")

        if predicted_et < 300:
            print(f"⚠️ Sensor {sensor_id}: Et is too low! ({predicted_et:.2f})")
        elif predicted_et > 500:
            print(f"⚠️ Sensor {sensor_id}: Et is too high! ({predicted_et:.2f})")

        if predicted_ev >= 1000:
            print(f"⚠️ Sensor {sensor_id}: Ev exceeds limit! ({predicted_ev:.2f})")

print("\n=== Pareto Front Optimization Completed ===")
