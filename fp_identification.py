import pandas as pd


df = pd.read_csv("data/beth_predictions.csv")


def classify(row):
    pred_anomaly = row['pred_label'] == -1
    truly_evil = row['evil'] == 1
    if pred_anomaly and truly_evil: return 'True Positive'
    if pred_anomaly and not truly_evil: return 'False Positive'
    if not pred_anomaly and truly_evil: return 'False Negative'
    return 'True Negative'


df['classification'] = df.apply(classify, axis=1)
# Summary
print(df['classification'].value_counts())

# Extract only False Positives

fp_df = df[df['classification'] == 'False Positive'].copy()
print(f"\nFalse Positives found: {len(fp_df)}")
print(f"FP rate: {len(fp_df)/len(df)*100:.1f}%)")

fp_df.to_csv("data/false_positives.csv", index=False)

#For visualisation purpouses (not necessary)
import matplotlib.pyplot as plt
counts = df['classification'].value_counts()
counts.plot(kind='bar', color=['green','red','orange','gray'])
plt.title("Classification results")
plt.tight_layout()
plt.savefig("outputs/classification_counts.png")