import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv('/Users/shijunshen/Documents/Environment/My Tableau Repository/Datasources/SmartFarmJournal/temperature_comparison.csv')

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot actual temp (black dotted line)
plt.plot(df['day'], df['temp_actu'], color='black', linestyle=':', label='Actual temp', linewidth=2)

# Plot predicted temp with best prediction (blue solid line)
plt.plot(df['day'], df['with_best_temp_pred'], color='blue', label='Best Prediction', linewidth=2)

# Plot predicted temp without best prediction (red solid line)
plt.plot(df['day'], df['no_best_temp_pred'], color='red', label='No Best Prediction', linewidth=2)

# Add labels and title
# plt.title('temp Prediction vs Actual', fontsize=36)
plt.xlabel('Day', fontsize=36)
plt.ylabel('Temp', fontsize=36)

# Set font size for ticks
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)

# Remove the legend
# plt.legend(fontsize=36)  # Commented out to remove the legend

# Display the plot
plt.tight_layout()  # Adjusts the plot to ensure everything fits
plt.show()