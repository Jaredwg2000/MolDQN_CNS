MolDQN-CNS Readme.

The code for this project is a bit of a mess. There are duplicates of some files that are required due to python imports, there are duplicates because I was trying to be organized and ended up copying myself, and everything is named pretty poorly.

To use this, set up a conda environment using environment.yml. To run the model, navigate to the "mol_dqn/chemgraph" folder, and run:

IMPORTANT - two copies of the mol2vec pretrained model, found here: https://deepchemdata.s3-us-west-1.amazonaws.com/trained_models/mol2vec_model_300dim.tar.gz are required. One should be put in the "project_data/mol2vec/mol2vec/models" folder, and the other should be placed in "mol_dqn/chemgraph/mol2vec/mol2vec/models". The model file is too big to push to github and I'm not experienced enough with it to consider a better workaround.

python multi_obj_opt.py --model_dir=${OUTPUT_DIR} --hparams="./configs/multi_obj_dqn.json" --start_molecule="CN1C(=CC2=C(C1=O)C(=NO2)C3=CC=NC=C3)C4=CC=C(C=C4)OC" --target_molecule="CN1C(=CC2=C(C1=O)C(=NO2)C3=CC=NC=C3)C4=CC=C(C=C4)OC" --similarity_weight=0.0

This will run the generative model on the supplied starting molecule, and output all of the molecules generated into a new csv file.

All of the data used in my project has been moved to the "project_data" folder. The names apporximately relate to different model parameters used to generate that data. 

There are two places with settings to fine-tune the model as it is, parameters can be modified in the "mol_dqn/chemgraph/configs/multi_obj_dqn.json" file, or the equation that the model tries to optimise for can be found in "mol_dqn/chemgraph/multi_obj_opt.py".

The models used to predict CNSMPO, VINA docking score and tHalf can be found in the "project_data" folder as notebooks. Running through each of these notbooks will train and save a new predictive model. This folder also contains other notebooks for analyzing any data, and comparing the models. Each contains similar but slightly different analyses.
