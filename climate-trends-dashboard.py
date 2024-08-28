import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

# Basic plot settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Load and prepare the data
url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
df = pd.read_csv(url, skiprows=1)
df['Year'] = df['Year'].astype(int)
df['Annual'] = pd.to_numeric(df['J-D'], errors='coerce')
df = df.dropna(subset=['Annual'])
df['Moving_Average'] = df['Annual'].rolling(window=5).mean()
df['Decade'] = (df['Year'] // 10) * 10

def plot_global_temp(ax):
    ax.plot(df['Year'], df['Annual'], label='Annual mean')
    ax.plot(df['Year'], df['Moving_Average'], label='5-year moving average', color='red')
    ax.set_title('Global Land-Ocean Temperature Index', fontsize=16, pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Temperature Anomaly (°C)', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)

def analyze_by_decade(ax):
    bar_width = 8
    decade_avg = df.groupby('Decade')['Annual'].mean().reset_index()
    
    # Add light grid lines with lower zorder
    ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
    
    # Create the bar plot with a zorder higher than the grid but lower than annotations
    bars = ax.bar(decade_avg['Decade'], decade_avg['Annual'], width=bar_width, 
                  color='skyblue', edgecolor='navy', zorder=2)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}°C',
                ha='center', va='bottom', fontsize=10, zorder=3)
    
    # Customize the plot
    ax.set_title('Average Temperature Anomaly by Decade', fontsize=16, pad=20)
    ax.set_xlabel('Decade', fontsize=12)
    ax.set_ylabel('Temperature Anomaly (°C)', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    # Set x-axis limits and add some padding
    ax.set_xlim(decade_avg['Decade'].min() - 5, decade_avg['Decade'].max() + 5)
    
    # Add a horizontal line at y=0 for reference
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
    
    # Highlight the warmest and coolest decades with higher zorder
    warmest_decade = decade_avg.loc[decade_avg['Annual'].idxmax()]
    coolest_decade = decade_avg.loc[decade_avg['Annual'].idxmin()]
    
    ax.annotate(f"Warmest: {warmest_decade['Annual']:.2f}°C",
                xy=(warmest_decade['Decade'], warmest_decade['Annual']),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                zorder=5)
    
    ax.annotate(f"Coolest: {coolest_decade['Annual']:.2f}°C",
                xy=(coolest_decade['Decade'], coolest_decade['Annual']),
                xytext=(0, -10), textcoords='offset points',
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                zorder=5)

def predict_future_temp(ax, years_ahead=30):
    X = df['Year'].values.reshape(-1, 1)
    y = df['Annual'].values
    model = LinearRegression()
    model.fit(X, y)
    future_years = np.array(range(df['Year'].max() + 1, df['Year'].max() + years_ahead + 1)).reshape(-1, 1)
    future_temps = model.predict(future_years)
    
    ax.scatter(df['Year'], df['Annual'], alpha=0.5, label='Historical data')
    ax.plot(df['Year'], model.predict(X), color='red', label='Linear trend')
    ax.scatter(future_years, future_temps, color='green', label='Predicted temperatures')
    ax.set_title('Temperature Anomaly: Historical Data and Future Prediction', fontsize=16, pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Temperature Anomaly (°C)', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)

def create_dashboard():
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Global Temperature Trends Dashboard', fontsize=24, y=0.98)
    
    plot_global_temp(axs[0, 0])
    analyze_by_decade(axs[0, 1])
    predict_future_temp(axs[1, 0])
    
    # Expanded Statistical analysis and key findings
    recent_temp = df[df['Year'] >= 1970]['Annual']
    historical_temp = df[df['Year'] < 1970]['Annual']
    t_stat, p_value = stats.ttest_ind(recent_temp, historical_temp)
    
    # Calculate additional statistics
    temp_change_rate = np.polyfit(df['Year'], df['Annual'], 1)[0]
    total_change = df['Annual'].iloc[-1] - df['Annual'].iloc[0]
    recent_trend = np.polyfit(df[df['Year'] >= 1970]['Year'], df[df['Year'] >= 1970]['Annual'], 1)[0]
    
    stats_text = (
        f"Key Findings and Statistical Analysis:\n\n"
        f"1. Overall Trend:\n"
        f"   • Mean temp anomaly (1880-2023): {df['Annual'].mean():.2f}°C\n"
        f"   • Total temperature change: {total_change:.2f}°C\n"
        f"   • Rate of change: {temp_change_rate:.4f}°C/year\n\n"
        f"2. Extreme Values:\n"
        f"   • Maximum anomaly: {df['Annual'].max():.2f}°C in {df.loc[df['Annual'].idxmax(), 'Year']}\n"
        f"   • Minimum anomaly: {df['Annual'].min():.2f}°C in {df.loc[df['Annual'].idxmin(), 'Year']}\n\n"
        f"3. Recent vs Historical Comparison:\n"
        f"   • Recent mean (1970-2023): {recent_temp.mean():.2f}°C\n"
        f"   • Historical mean (1880-1969): {historical_temp.mean():.2f}°C\n"
        f"   • T-test results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}\n"
        f"   • Recent trend (1970-2023): {recent_trend:.4f}°C/year\n\n"
        f"4. Decadal Analysis:\n"
        f"   • Warmest decade: {df.groupby('Decade')['Annual'].mean().idxmax()}s\n"
        f"   • Coolest decade: {df.groupby('Decade')['Annual'].mean().idxmin()}s\n\n"
        f"5. Considerations:\n"
        f"   • The data shows a clear warming trend, especially pronounced in recent decades.\n"
        f"   • The rate of warming has accelerated since 1970.\n"
        f"   • The p-value (< 0.05) indicates a statistically significant difference between\n"
        f"     recent and historical temperatures.\n"
        f"   • While natural variability exists, the observed trends align with scientific\n"
        f"     consensus on human-induced climate change.\n"
        f"   • Future projections suggest continued warming, emphasizing the need for\n"
        f"     climate action and adaptation strategies."
    )
    axs[1, 1].text(0.05, 0.95, stats_text, transform=axs[1, 1].transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    axs[1, 1].set_title('Key Findings and Considerations', fontsize=16, pad=20)
    axs[1, 1].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)
    plt.savefig('global_temperature_trends_dashboard.png', dpi=300, bbox_inches='tight')
    print("Dashboard saved as 'global_temperature_trends_dashboard.png'")

# Create and save the dashboard
create_dashboard()

# Display summary statistics
print("\nSummary Statistics:")
print(df['Annual'].describe())

temp_change_rate = np.polyfit(df['Year'], df['Annual'], 1)[0]
print(f"\nRate of temperature change: {temp_change_rate:.4f}°C per year")

total_change = df['Annual'].iloc[-1] - df['Annual'].iloc[0]
print(f"Total temperature change from {df['Year'].iloc[0]} to {df['Year'].iloc[-1]}: {total_change:.2f}°C")
