# COVID Analytics App

A robust PySpark-based data analytics pipeline for processing, standardizing, and analyzing global COVID-19 datasets. The project demonstrates advanced data engineering practices and compliance with strict audit requirements.

## Features Implemented

### Data Engineering & Standardization
- **PySpark Pipelines:** Built scalable data pipelines using Apache Spark to ingest and process large-scale COVID-19 datasets.
- **Data Cleansing:** Standardized text and categorical data using robust string manipulation (`regexp_replace()`) and handled missing values across multiple global datasets.

### Advanced Analytics
- **Growth Metrics:** Calculated complex epidemiological metrics, including the death growth formula and infection rates, across different regions and timeframes.
- **Aggregations:** Performed deep data aggregations to generate reports on day-wise, country-wise, and US county-level trends.

### Visualizations & Reporting
- **Data Visualizations:** Configured clear, readable charts and graphs (handling axis labels and layouts) directly within the notebook to visualize pandemic trends.
- **Automated Outputs:** Exported finalized data pipelines and standardized reports to the `pipeline_output` directory for downstream use.

## Tech Stack
- Python 3
- Apache Spark / PySpark (Distributed Data Processing)
- Jupyter Notebooks
- Matplotlib / Seaborn (Visualizations)

## How to Run

1. Ensure you have **Apache Spark** and **PySpark** installed and configured on your system.
2. Open the `main.ipynb` Jupyter Notebook.
3. The raw datasets are already included in the `data` directory (e.g., `covid_19_clean_complete.csv`, `usa_county_wise.csv`).
4. Run all cells sequentially to trigger the data ingestion, execute the standardization logic, and generate the final visualization reports.
5. Processed datasets and reports will be automatically generated in the `pipeline_output` folder.
