All files in this folder regard data collection and treatment.

- data/:
    - deter_notification_data/: Contains original and treated .csv files of notified cases of deforestation in km² obtained from DETER ([https://terrabrasilis.dpi.inpe.br/app/dashboard/alerts/biomes/amazonia-nb/daily/#](https://terrabrasilis.dpi.inpe.br/app/dashboard/alerts/biomes/amazonia-nb/daily/#))
    - sivep_notification_data/: Contains original .parquet and treated .csv files of cases of infection by malaria obtained from SIVEP
    - TRAJETORIAS_DATASET_Epidemiological_dimension_indicators.csv: Contains epidemiological data regarding type of disease, time period, location, residential zone, number of cases and inidence from the Trajetorias project
    - TRAJETORIAS_DATASET_Population_indicators.csv: Contains populational data regarding location, population in different time periods and estimated population
    - climate_api_data_2016_2024.csv: Contains climate data obtained from the Mosqlimate datastore ([https://api.mosqlimate.org/api/docs#/datastore/datastore_api_get_copernicus_brasil](https://api.mosqlimate.org/api/docs#/datastore/datastore_api_get_copernicus_brasil))
    - ibge_manaus_fixed_rural_population_data_2016_2024.csv: Contains estimated daily rural and total population of Manaus, calculated in [../notebooks/Manaus_Population_Correction.ipynb](https://github.com/RaphaLevy/Study_on_malaria_dynamics_v2/blob/main/data_files/notebooks/Manaus_Population_Correction.ipynb)
    - ibge_manaus_population_data_2016_2024.csv: Contains interpolated daily population data of Manaus, obtained from IBGE
        - 2000: [https://biblioteca.ibge.gov.br/index.php/biblioteca-catalogo?view=detalhes&id=7308 (Tabelas Excel > AM > UF > Tabela 15)](https://biblioteca.ibge.gov.br/index.php/biblioteca-catalogo?view=detalhes&id=7308)
        - 2010: [https://www.ibge.gov.br/estatisticas/sociais/populacao/9662-censo-demografico-2010.html?=&t=resultados (Amazonas > Tabela 2.1.3)](https://www.ibge.gov.br/estatisticas/sociais/populacao/9662-censo-demografico-2010.html?=&t=resultados)
        - 2022, 2025: [https://cidades.ibge.gov.br/brasil/am/manaus/panorama](https://cidades.ibge.gov.br/brasil/am/manaus/panorama); [https://www.ibge.gov.br/cidades-e-estados/am/manaus.html](https://www.ibge.gov.br/cidades-e-estados/am/manaus.html)
    - inpe_fire_counts_data_2016_2024.csv: Contains daily fire counts data gathered by INPE, obtained from the Base dos Dados dataset
([https://basedosdados.org/dataset/f06f3cdc-b539-409b-b311-1ff8878fb8d9?table=a3696dc2-4dd1-4f7e-9769-6aa16a1556b8](https://basedosdados.org/dataset/f06f3cdc-b539-409b-b311-1ff8878fb8d9?table=a3696dc2-4dd1-4f7e-9769-6aa16a1556b8))

- notebooks/:
    - DETER_Deforestation.ipynb: Collects and treats deforestation notification data from 2016 to 2024, obtained from the TerraBrasilis website
    - IBGE_Population_Data.ipynb: Uses and treats IBGE data to estimate yearly and daily population of Manaus from 2016 to 2024
    - INPE_Fire_Counts.ipynb: Collects and treats fire counts data from 2016 to 2024, obtained from the Base dos Dados dataset
    - Manaus_Population_Correction.ipynb: Collects an treats populational data from Manaus in different time periods, looking for possible inconsistencies on the data
    - Manaus_Vivax_Cases.ipynb: Gathers treated SIVEP data on malaria cases from 2003 to 2023 and filters them to Manaus 
    - Mosqlimate_API_Data.ipynb: Collects and treats climate data from 2016 to 2024, obtained from the Mosqlimate datastore 
    - Parquet_to_CSV.ipynb: Converts original .parquet files to .csv
    - SIVEP_CSV_Data_Cleaning.ipynb: Treats converted .csv files by removing unused columns and filtering to specific cases
    - TRAJETORIAS_Data_Cleaning.ipynb: Collects and treats epidemiological and populational data gathered from the Trajetorias Project ([https://www.nature.com/articles/s41597-023-01962-1](https://www.nature.com/articles/s41597-023-01962-1) (Paper); [https://zenodo.org/records/7098053](https://zenodo.org/records/7098053) (Dataset))

