All files in this folder regard models in development. They are separated in SIR-SEI (original formulation) and SEIRS-SEI (updated formulation)

- sir_sei:
  - Original_Model_PM_Parameters.ipynb: Base SIR/SEI model, based on the original parameters of Parham & Michael (2010), with minor modifications (DOI: 10.1007/978-1-4419-6064-1_13)
  - Original_Model_Dissertation_Parameters.ipynb: Base SIR/SEI model, based on the parameters used in the development of the project during the Undergraduate Dissertation [(https://github.com/RaphaLevy/Undergraduate_Dissertation)](https://github.com/RaphaLevy/Undergraduate_Dissertation)
  - Plot_Model_Base_Functions.ipynb: File for plotting environmental functions used in the model using DDE methods (**Under development**)
  - Test_DDE_Model.ipynb: Updated SIR/SEI model to use DDE methods and appropriate solver, with analysis of infection dynamics in relation to climate factors
- seirs_sei:
  - Test_SEIRS_SEI_Model.ipynb: SEIRS/SEI model with DDE solver, comparing infection data and result of model
  - Test_SEIRS_SEI_Model_Intervention.ipynb: Updated SEIRS/SEI model with DDE solver, including intervention dynamics to approximate model result to actual data
