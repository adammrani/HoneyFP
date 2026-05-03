import pandas as pd
import numpy as np

fp_df = pd.read_csv("data/false_positives.csv")
print(fp_df.columns.tolist())
def auto_label(row):
    if row['userId'] == 0 and row['returnValue'] == 0:
        return 'benign'
    if row['userId'] == 0 and row['returnValue'] < 0:
        return 'suspicious'
    if row['sus'] == 0:
        return 'misclassified'
    return 'benign' #default

fp_df['fp_label'] = fp_df.apply(auto_label, axis=1)

print(fp_df['fp_label'].value_counts())

def describe_event(row):
    return (
        f"Event {row['eventId']} ({row.get('eventName','unknown')}), "
        f"processId={row['processId']}, userId={row['userId']}, "
        f"returnValue={row['returnValue']}, score={row['anomaly_score']:.3f}"
    )

fp_df['description'] = fp_df.apply(describe_event, axis=1)
fp_df.to_csv("data/false_positives_labeled.csv", index=False)
print("Saved false_positives_labeled.csv")