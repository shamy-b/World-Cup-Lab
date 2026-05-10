import pandas as pd
df = pd.read_csv('dataset/results.csv')
df['date'] = pd.to_datetime(df['date'])
test = df[(df['date'] >= '2018-01-01') & df['home_score'].notna()]
outcomes = test.apply(lambda r: 2 if r.home_score > r.away_score else (0 if r.home_score < r.away_score else 1), axis=1)
print("Test set class distribution (0=Away, 1=Draw, 2=Home):")
print(outcomes.value_counts(normalize=True).sort_index())
print()
print("Total test matches:", len(test))
print()
# Also check neutral venue stats
print("Neutral venue %:", test['neutral'].mean())
print()
# Check tournament distribution in test set
print("Top tournaments in test set:")
print(test['tournament'].value_counts().head(10))
