from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

spark = SparkSession.builder.appName("COVID_Analytics").getOrCreate()

# 1. Data Loading & Schema Handling

dfs = {
    "full_grouped": spark.read.csv("data/raw/full_grouped.csv", header=True, inferSchema=True),
    "clean_complete": spark.read.csv("data/raw/covid_19_clean_complete.csv", header=True, inferSchema=True),
    "country_latest": spark.read.csv("data/raw/country_wise_latest.csv", header=True, inferSchema=True),
    "day_wise": spark.read.csv("data/raw/day_wise.csv", header=True, inferSchema=True),
    "usa_county": spark.read.csv("data/raw/usa_county_wise.csv", header=True, inferSchema=True),
    "worldometer": spark.read.csv("data/raw/worldometer_data.csv", header=True, inferSchema=True)
}

for name, df in dfs.items():
    df.printSchema()
    print(df.count())


# 2. Data Cleaning Tasks
dfs["clean_complete"].filter(F.col("Province/State").isNull()).groupBy("Country/Region").count().withColumnRenamed("count", "Null_Count").show()
dfs["clean_complete"] = dfs["clean_complete"].fillna({"Province/State": "Unknown"})


# 3. Standardize Country Names
for k in ["full_grouped", "country_latest", "worldometer"]:
    dfs[k] = dfs[k].withColumn("Country/Region", F.regexp_replace(F.regexp_replace(F.col("Country/Region"), "^US$", "USA"), "^Korea$", "South Korea"))

# 4. Remove Duplicate Daily Records
dfs["full_grouped"] = dfs["full_grouped"].dropDuplicates(["Country/Region", "Date"])

# 5. Top 10 Countries by Total Confirmed Cases
top_cases = dfs["country_latest"].select("Country/Region", "Confirmed").orderBy(F.col("Confirmed").desc()).limit(10).toPandas()

# 6. Top 10 Countries by Death Rate
top_deaths = dfs["country_latest"].select("Country/Region", "Deaths / 100 Cases").orderBy(F.col("Deaths / 100 Cases").desc()).limit(10).toPandas()

# 7. WHO Region-wise Total Cases
who_summary = dfs["full_grouped"].groupBy("WHO Region").agg(F.sum("Confirmed").alias("Total_Cases"), F.sum("Deaths").alias("Total_Deaths"), F.sum("Recovered").alias("Total_Recovered")).toPandas()

print("Top 10 Countries by Total Confirmed:")
print(top_cases.to_string(index=False))
print("\n")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].bar(top_cases["Country/Region"], top_cases["Confirmed"])
axes[0].set_title("Top 10 Confirmed Cases")
axes[0].tick_params(axis='x', rotation=90)

axes[1].barh(top_deaths["Country/Region"], top_deaths["Deaths / 100 Cases"])
axes[1].set_title("Top 10 Death Rates")

axes[2].pie(who_summary["Total_Cases"], labels=who_summary["WHO Region"], autopct='%1.1f%%')
axes[2].set_title("Cases by WHO Region")

plt.tight_layout()
plt.show()

# 8. Daily Global New Cases Trend
daily = dfs["day_wise"].orderBy("Date").toPandas()

# 9. Daily Global Death Growth Trend
death_growth = dfs["day_wise"].withColumn("Prev_Deaths", F.lag("New deaths").over(Window.orderBy("Date"))).withColumn("Death_Growth", F.when((F.col("Prev_Deaths").isNull()) | (F.col("Prev_Deaths") == 0), 0).otherwise((F.col("New deaths") / F.col("Prev_Deaths")) * 100)).toPandas()

# 10. Monthly COVID Case Growth
monthly = dfs["full_grouped"].withColumn("Month", F.month("Date")).groupBy("Month").agg(F.sum("Confirmed").alias("Monthly_Cases")).orderBy("Month").toPandas()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].plot(pd.to_datetime(daily["Date"]), daily["New cases"])
axes[0].set_title("Daily Global New Cases")
axes[0].tick_params(axis='x', rotation=45)

axes[1].plot(pd.to_datetime(death_growth["Date"]), death_growth["Death_Growth"])
axes[1].set_title("Daily Death Growth (%)")
axes[1].tick_params(axis='x', rotation=45)

axes[2].plot(monthly["Month"], monthly["Monthly_Cases"], marker='o')
axes[2].set_title("Monthly Case Growth")
axes[2].set_xticks(monthly["Month"])

plt.tight_layout()
plt.show()

# 11. Top 5 Most Affected Countries Per WHO Region
top_5_region = dfs["country_latest"].withColumn("Rank", F.dense_rank().over(Window.partitionBy("WHO Region").orderBy(F.col("Confirmed").desc()))).filter(F.col("Rank") <= 5).toPandas()

# 12. Country-wise Daily Case Increase
country_increase = dfs["full_grouped"].withColumn("Daily_Increase", F.col("Confirmed") - F.lag("Confirmed").over(Window.partitionBy("Country/Region").orderBy("Date")))

usa_trend = country_increase.filter(F.col("Country/Region") == "USA").toPandas()

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
top_5_region.pivot(index="Country/Region", columns="WHO Region", values="Confirmed").plot(kind='bar', ax=axes[0], width=0.8)
axes[0].set_title("Top 5 Countries by WHO Region")
axes[0].tick_params(axis='x', rotation=90)
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

axes[1].plot(pd.to_datetime(usa_trend["Date"]), usa_trend["Daily_Increase"])
axes[1].set_title("USA Daily Case Increase Trend")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 13. Compare Latest Dataset Sources
mismatches = dfs["country_latest"].alias("c").join(dfs["worldometer"].alias("w"), "Country/Region").select("Country/Region", F.abs(F.col("c.Confirmed") - F.col("w.TotalCases")).alias("Confirmed_Diff"), F.abs(F.col("c.Deaths") - F.col("w.TotalDeaths")).alias("Death_Diff"), F.abs(F.col("c.Recovered") - F.col("w.TotalRecovered")).alias("Recovery_Diff")).filter(F.col("Confirmed_Diff") > 10000)

# 14. Population vs Total Cases
infection_rates = dfs["worldometer"].filter(F.col("Population") > 0).withColumn("Infection_Rate", (F.col("TotalCases") / F.col("Population")) * 100).orderBy(F.col("Infection_Rate").desc()).limit(20).toPandas()

print("Countries With Large Mismatches (>10,000):")
mismatches.show()

plt.figure(figsize=(12, 5))
plt.scatter(infection_rates["Country/Region"], infection_rates["Infection_Rate"], color='red')
plt.xticks(rotation=90)
plt.title("Top 20 Countries by Infection Rate (%)")
plt.show()

# 15. USA State-wise Case Distribution
usa_states = dfs["usa_county"].groupBy("Province_State").count().orderBy(F.col("count").desc()).limit(20).toPandas()

# 16.  Latitude-Longitude Based Case Clusters
geo_data = dfs["clean_complete"].select("Lat", "Long", "Confirmed").filter(F.col("Lat").isNotNull() & F.col("Long").isNotNull() & (F.col("Confirmed") > 0)).toPandas()

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
axes[0].bar(usa_states["Province_State"], usa_states["count"])
axes[0].set_title("Top 20 USA States by County Count")
axes[0].tick_params(axis='x', rotation=90)

axes[1].scatter(geo_data["Long"], geo_data["Lat"], s=geo_data["Confirmed"]/10000, alpha=0.5)
axes[1].set_title("Global Case Clusters (Lat/Long)")

plt.tight_layout()
plt.show()

# 17. Recovery Rate Analysis
recovery = dfs["country_latest"].filter(F.col("Confirmed") > 1000).withColumn("Rec_Rate", (F.col("Recovered") / F.col("Confirmed")) * 100)

# 18. Active Case Burden Analysis
best_rec = recovery.orderBy(F.col("Rec_Rate").desc()).limit(10).toPandas()
worst_rec = recovery.orderBy(F.col("Rec_Rate").asc()).limit(10).toPandas()
high_risk = dfs["country_latest"].filter(F.col("Active") > F.col("Recovered")).select("Country/Region", "Active", "Recovered")

# 19. Identify Pandemic Peaks
peaks = dfs["day_wise"].agg(F.max("New cases").alias("Max_Cases"), F.max("New deaths").alias("Max_Deaths")).collect()[0]
peak_dates = dfs["day_wise"].filter((F.col("New cases") == peaks["Max_Cases"]) | (F.col("New deaths") == peaks["Max_Deaths"])).toPandas()

print("High-Risk Countries (Active Cases > Recovered):")
high_risk.show(10)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].bar(best_rec["Country/Region"], best_rec["Rec_Rate"])
axes[0].tick_params(axis='x', rotation=90)
axes[0].set_title("Best Recovery Rates")

axes[1].bar(worst_rec["Country/Region"], worst_rec["Rec_Rate"], color='orange')
axes[1].tick_params(axis='x', rotation=90)
axes[1].set_title("Worst Recovery Rates")

axes[2].plot(pd.to_datetime(daily["Date"]), daily["New cases"], label="Cases")
axes[2].scatter(pd.to_datetime(peak_dates["Date"]), peak_dates["New cases"], color='red', s=100, label="Peak Marker", zorder=5)
axes[2].set_title("Global Cases with Peaks")
axes[2].tick_params(axis='x', rotation=45)
axes[2].legend()

plt.tight_layout()
plt.show()

# 20. Create Severity Category
severity = dfs["country_latest"].withColumn("Severity_Category", F.when(F.col("Confirmed") < 10000, "Low").when((F.col("Confirmed") >= 10000) & (F.col("Confirmed") <= 100000), "Medium").when((F.col("Confirmed") > 100000) & (F.col("Confirmed") <= 1000000), "High").otherwise("Critical"))

sev_counts = severity.groupBy("Severity_Category").count().toPandas()
plt.figure(figsize=(6, 6))
plt.pie(sev_counts["count"], labels=sev_counts["Severity_Category"], autopct='%1.1f%%')
plt.title("Global Severity Category Distribution")
plt.show()

# 21. Build COVID Analytics Pipeline
import os

os.makedirs("pipeline_output", exist_ok=True)

try:
    severity.toPandas().to_parquet("pipeline_output/severity_categories.parquet", index=False)
    who_summary.to_csv("pipeline_output/region_summary.csv", index=False)
    print("Output files successfully written to 'pipeline_output/' directory.\n")
except Exception as e:
    print(f"Could not write output files: {e}\n")

print("Final Report Insights:")
print(f"Highest Death Rate: {top_deaths.iloc[0]['Country/Region']}")
print(f"Fastest Recovering Region: {who_summary.sort_values('Total_Recovered', ascending=False).iloc[0]['WHO Region']}")
print(f"Global Cases Peaked On: {peak_dates[peak_dates['New cases'] == peaks['Max_Cases']]['Date'].values[0]}")
print(f"Global Deaths Peaked On: {peak_dates[peak_dates['New deaths'] == peaks['Max_Deaths']]['Date'].values[0]}")
print(f"Best Recovery Handling Country: {best_rec.iloc[0]['Country/Region']}")