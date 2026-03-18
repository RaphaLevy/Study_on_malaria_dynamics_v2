All files in this folder regard data collection and treatment.

- data/:
    - deter_notification_data/: Contains original and treated .csv files of notified cases of deforestation in km² obtained from DETER [https://terrabrasilis.dpi.inpe.br/app/dashboard/alerts/biomes/amazonia-nb/daily/#](https://terrabrasilis.dpi.inpe.br/app/dashboard/alerts/biomes/amazonia-nb/daily/#)
    - sivep_notification_data/: Contains original .parquet and treated .csv files of cases of infection by malaria obtained from SIVEP
    - climate_api_data_2016_2024.csv: Contains climate data obtained from the Mosqlimate datastore [https://api.mosqlimate.org/api/docs#/datastore/datastore_api_get_copernicus_brasil](https://api.mosqlimate.org/api/docs#/datastore/datastore_api_get_copernicus_brasil)
    - inpe_fire_counts_data_2016_2024.csv: Contains daily fire counts data gathered by INPE, obtained from the Base dos Dados dataset
[https://basedosdados.org/dataset/f06f3cdc-b539-409b-b311-1ff8878fb8d9?table=a3696dc2-4dd1-4f7e-9769-6aa16a1556b8](https://basedosdados.org/dataset/f06f3cdc-b539-409b-b311-1ff8878fb8d9?table=a3696dc2-4dd1-4f7e-9769-6aa16a1556b8)

- notebooks/:
    - DETER_Deforestation.ipynb: Collects and treats deforestation notification data from 2016 to 2024, obtained from the TerraBrasilis website
    - INPE_Fire_Counts.ipynb: Collects and treats fire counts data from 2016 to 2024, obtained from the Base dos Dados dataset
    - Mosqlimate_API_Data.ipynb: Collects and treats climate data from 2016 to 2024, obtained from the Mosqlimate datastore 
    - Parquet_to_CSV.ipynb: Converts original .parquet files to .csv
    - SIVEP_CSV_Data_Cleaning.ipynb: Treats converted .csv files by removing unused columns and filtering to specific cases

