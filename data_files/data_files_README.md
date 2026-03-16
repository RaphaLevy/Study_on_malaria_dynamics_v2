All files in this folder regard data collection and treatment.

- data/:
    - climate_api_data_2016_2024.csv: Contains climate data obtained from the Mosqlimate datastore [https://api.mosqlimate.org/api/docs#/datastore/datastore_api_get_copernicus_brasil](https://api.mosqlimate.org/api/docs#/datastore/datastore_api_get_copernicus_brasil)
    - sivep_notification_data/: Contains original .parquet and treated .csv files of cases of infection by malaria obtained from SIVEP

- notebooks/:
    - Mosqlimate_API_Data.ipynb: Collects and treats climate data from 2016 to 2024, obtained from the Mosqlimate datastore 
    - Parquet_to_CSV.ipynb: Converts original .parquet files to .csv
    - SIVEP_CSV_Data_Cleaning.ipynb: Treats converted .csv files by removing unused columns and filtering to specific cases
