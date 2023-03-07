MolDQN-CNS Readme.

Quick Start:
1. Set up a conda environment using the supplied environment.yml file.

2. Add a copy of the pretrained mol2vec model, found at https://deepchemdata.s3-us-west-1.amazonaws.com/trained_models/mol2vec_model_300dim.tar.gz to the folders "project_data/mol2vec/mol2vec/models" and "mol_dqn/chemgraph/mol2vec/mol2vec/models".

3. Run the generative model by navigating to "mol_dqn/chemgraph" and using the following:

python multi_obj_opt.py --model_dir=${OUTPUT_DIR} --hparams="./configs/multi_obj_dqn.json" --start_molecule="CN1C(=CC2=C(C1=O)C(=NO2)C3=CC=NC=C3)C4=CC=C(C=C4)OC" --target_molecule="CN1C(=CC2=C(C1=O)C(=NO2)C3=CC=NC=C3)C4=CC=C(C=C4)OC" --similarity_weight=0.0

This will generate an output csv file, containing all of the molecules the model generated as it trained.

Other information

All of the data used in my project has been moved to the "project_data" folder. The names apporximately relate to different model parameters used to generate that data. 

There are two places with settings to fine-tune the model as it is, parameters can be modified in the "mol_dqn/chemgraph/configs/multi_obj_dqn.json" file, or the equation that the model tries to optimise for can be found in "mol_dqn/chemgraph/multi_obj_opt.py".

The models used to predict CNSMPO, VINA docking score and tHalf can be found in the "project_data" folder as notebooks. Running through each of these notbooks will train and save a new predictive model. This folder also contains other notebooks for analyzing any data, and comparing the models. The notebook "RandomForests" contains the code used to generate and train the models used in comparison to the dense neural network - Random Forests, Decision Trees and Linear Regressions. The "Plots" notebook contains code to parse the data, including drawing selected sets of molecules and producing various graphs of the data. The "RunModels" notebooks was used for running the predictive models on outputs from the generative model, and also to produce some graphs. This was needed to generate data that was not recorded in the outputs of early model runs or in the original data. The "SMILESLIST" notebook was used to clean up the original datasets such that they could be used by the generative and predictive models.



