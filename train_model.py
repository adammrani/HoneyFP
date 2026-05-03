import pandas as pd
import joblib
import glob
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURES = ['processId', 'threadId', 'parentProcessId', 
            'userId', 'mountNamespace', 'eventId', 
            'argsNum', 'returnValue']

path = './dataset'
all_files = glob.glob(os.path.join(path, "*.csv"))

ind_df_list = []

for filename in all_files:
    df = pd.read_csv(filename, header=0)
    #Just to know from which csv file the column came from:
    df['source_file'] = os.path.basename(filename)
    ind_df_list.append(df)

#Concatenate all the dfs
beth_df = pd.concat(ind_df_list, axis=0, ignore_index=True)


# Train normal samples (evil=0)

normal = beth_df[beth_df['evil'] == 0]
X_train = normal[FEATURES]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Isolation forest : contamination = expected % of anomalies

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
model.fit(X_train_scaled)

#Predict on FULL dataset
X_all_scaled = scaler.transform(beth_df[FEATURES])
# Add predicted label column
beth_df['pred_label'] = model.predict(X_all_scaled)
beth_df['anomaly_score'] = -model.score_samples(X_all_scaled) #applying negation
#Save models and predictions

joblib.dump(model, "models/isolation_forest.pkl")
joblib.dump(scaler, "models/scaler.pkl")

beth_df.to_csv("data/beth_predictions.csv", index=False)
print("Model saved. Predictions Saved.")


#pred_label: 1 = model says NORMAL, -1 = model says ANOMALY
#evil:  0 = truly normal, 1= truly anomalous