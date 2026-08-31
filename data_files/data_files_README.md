All files in this folder regard data collection and treatment.

- data/:
    - deter_notification_data/: Contains original and treated .csv files of notified cases of deforestation in km² obtained from DETER ([TerraBrasilis - DETER (notices)](https://terrabrasilis.dpi.inpe.br/app/dashboard/alerts/biomes/amazonia-nb/daily/#))
    - sivep_notification_data/: Contains original .parquet and treated .csv files of cases of infection by malaria obtained from SIVEP
    - TRAJETORIAS_DATASET_Epidemiological_dimension_indicators.csv: Contains epidemiological data regarding type of disease, time period, location, residential zone, number of cases and inidence from the Trajetorias project
    - TRAJETORIAS_DATASET_Population_indicators.csv: Contains populational data regarding location, population in different time periods and estimated population
    - climate_api_data_2016_2024.csv: Contains climate data obtained from the Mosqlimate datastore ([Mosqlimate API](https://api.mosqlimate.org/api/docs#/datastore/datastore_api_get_copernicus_brasil))
    - climate_api_data_aug_2003_jul_2004.csv: Contains climate data obtained from the Mosqlimate datastore, gathering data from August 2003 to July 2004, used in the estimation of proper parameters of the birth rate, based on the work by de Barros, Honório and Arruda (2011) ([Mosqlimate API](https://api.mosqlimate.org/api/docs#/datastore/datastore_api_get_copernicus_brasil))
    - ibge_manaus_fixed_rural_population_data_2016_2024.csv: Contains estimated daily rural and total population of Manaus, calculated in [../notebooks/Manaus_Population_Correction.ipynb](https://github.com/RaphaLevy/Study_on_malaria_dynamics_v2/blob/main/data_files/notebooks/Manaus_Population_Correction.ipynb)
    - ibge_manaus_population_data_2016_2024.csv: Contains interpolated daily population data of Manaus, obtained from IBGE
        - 2000: [IBGE 2000](https://biblioteca.ibge.gov.br/index.php/biblioteca-catalogo?view=detalhes&id=7308)
        - 2010: [IBGE 2010](https://www.ibge.gov.br/estatisticas/sociais/populacao/9662-censo-demografico-2010.html?=&t=resultados)
        - 2022, 2025: [IBGE 2022](https://cidades.ibge.gov.br/brasil/am/manaus/panorama); [IBGE 2025](https://www.ibge.gov.br/cidades-e-estados/am/manaus.html)
    - inpe_fire_counts_data_2016_2024.csv: Contains daily fire counts data gathered by INPE, obtained from the Base dos Dados dataset
([Banco de Dados de Queimadas](https://basedosdados.org/dataset/f06f3cdc-b539-409b-b311-1ff8878fb8d9?table=a3696dc2-4dd1-4f7e-9769-6aa16a1556b8))
    - mapbiomas_fire_tendencies_1985_2025.csv: Contains monthly tendencies of burnt out areas in hectares, from 1985 to 2025, obtained from MapBiomas ([Mapbiomas - Monthly trends by year](https://plataforma.brasil.mapbiomas.org/fire/fire_monthly?activeBaseMap=1&layersOpacity=100&activeModule=fire&activeModuleContent=fire:fire_monthly&activeYear=2023&mapPosition=-2.573422,-59.981338,9&timelineLimitsRange=1985,2023&baseParams[territoryType]=4&baseParams[territory]=146&baseParams[territories]=146;1302603%20-%20Manaus%20(AM);4;Munic%C3%ADpio;-3.22220882399995;-60.802727218;-1.92430443799998;-59.1599483219999&baseParams[activeClassTreeOptionValue]=fire_monthly&baseParams[activeClassTreeNodeIds]=603,604,605,606,607,608,609,610,611,612,613,614&baseParams[activeSubmodule]=fire_monthly&baseParams[yearRange]=1985-2023&tl[id]=32&tl[themeKey]=fire&tl[subthemeKey]=fire_monthly&tl[pixelValues][]=1&tl[pixelValues][]=2&tl[pixelValues][]=3&tl[pixelValues][]=4&tl[pixelValues][]=5&tl[pixelValues][]=6&tl[pixelValues][]=7&tl[pixelValues][]=8&tl[pixelValues][]=9&tl[pixelValues][]=10&tl[pixelValues][]=11&tl[pixelValues][]=12&tl[legendKey]=fire_monthly_mapbiomas_monthly&tl[year]=2025&t[regionKey]=brazil&t[ids][]=8edd8cc7-e345-4541-a26d-37551e3628fa&t[divisionCategoryId]=213))

- notebooks/:
    - Biting_Rate_Estimation.ipynb: Gathers and treats data related to gonotrophic cycle and biting rate, used to estimate D1, obtained from the works of de Barros, Honório and Arruda (2011)
    - DETER_Deforestation.ipynb: Collects and treats deforestation notification data from 2016 to 2024, obtained from the TerraBrasilis website
    - Environmental_Factor_Plots.ipynb: Plots temperature, precipitation and humidity histograms based on Mosqlimate data, in addition to humidity seasonality, comparison between survival functions (p(H)) and seasonal impacts on p(H)  
    - IBGE_Population_Data.ipynb: Uses and treats IBGE data to estimate yearly and daily population of Manaus from 2016 to 2024
    - INPE_Fire_Counts.ipynb: Collects and treats fire counts data from 2016 to 2024, obtained from the Base dos Dados dataset
    - Manaus_Population_Correction.ipynb: Collects an treats populational data from Manaus in different time periods, looking for possible inconsistencies on the data
    - Manaus_Vivax_Cases.ipynb: Gathers treated SIVEP data on malaria cases from 2003 to 2023 and filters them to Manaus 
    - Mosqlimate_API_Data.ipynb: Collects and treats climate data from 2016 to 2024, obtained from the Mosqlimate datastore 
    - Parquet_to_CSV.ipynb: Converts original .parquet files to .csv
    - SIVEP_CSV_Data_Cleaning.ipynb: Treats converted .csv files by removing unused columns and filtering to specific cases
    - TRAJETORIAS_Data_Cleaning.ipynb: Collects and treats epidemiological and populational data gathered from the Trajetorias Project ([https://www.nature.com/articles/s41597-023-01962-1](https://www.nature.com/articles/s41597-023-01962-1) (Paper); [https://zenodo.org/records/7098053](https://zenodo.org/records/7098053) (Dataset))

