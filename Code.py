import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Read the CSV file into a DataFrame
df = pd.read_csv('genotype_data.csv')

# Display the first few rows of the DataFrame
print("DataFrame Preview:")
print(df.head())

# List of traits
traits = ['PH_SSI', 'PH_STI', 'RL_SSI', 'RL_STI', 'RDW_SSI', 'RDW_STI', 'SDW_SSI', 'SDW_STI']

# Iterate over each trait
for trait in traits:
    print(f"\nAnalysis for {trait}:")

    # Perform one-way ANOVA to test for differences in yield among origins
    origins = df['Origin'].unique()
    origin_groups = [df[trait][df['Origin'] == origin] for origin in origins]

    # One-way ANOVA
    anova_result = f_oneway(*origin_groups)

    # Display ANOVA results
    print("\nANOVA Results:")
    print("F-statistic:", anova_result.statistic)
    print("P-value:", anova_result.pvalue)

    # Interpret the results
    alpha = 0.05
    if anova_result.pvalue < alpha:
        print(
            f"\nReject the null hypothesis: There is significant evidence that at least one origin differs in terms of {trait}.")

        # Perform Tukey's HSD post hoc tests
        tukey_results = pairwise_tukeyhsd(df[trait], df['Origin'])
        print("\nTukey's HSD Results:")
        print(tukey_results)

        # Export Tukey's HSD Results to CSV
        tukey_results_df = pd.DataFrame(data=tukey_results._results_table.data[1:],
                                        columns=tukey_results._results_table.data[0])
        tukey_results_df.to_csv(f'tukey_hsd_results_{trait}.csv', index=False)
        print(f"\nTukey's HSD Results exported to 'tukey_hsd_results_{trait}.csv'.")

    else:
        print(
            f"\nFail to reject the null hypothesis: There is no significant evidence that origins differ in terms of {trait}.")

    # Visualize the results with boxplots
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='Origin', y=trait, data=df, palette='Set3', hue='Origin', legend=False)
    plt.title(f'{trait} Comparison by Origin')
    plt.xlabel('Origin')
    plt.ylabel(trait)

    # Annotate boxplots with average values
    # Calculate average values for each origin
    average_values = df.groupby('Origin')[trait].mean().to_dict()
    for i, origin in enumerate(origins):
        mean_value = average_values[origin]
        ax.text(i, mean_value, f'{mean_value:.2f}', color='black', ha="center")

    plt.show()
