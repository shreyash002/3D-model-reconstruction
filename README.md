# 3D-model-reconstruction

3D reconstruction from input images.



Files:
- Evaluation.ipynb : Evaluation notebook which loads a given model and runs the validation set to get the Average IoU score for the model.
- Load_Matlab.py : Used to visualize the binvox file in Matlab. Not used anymore since we use another script for this currently.
- binvox_rw.py : Used to visualize binvox file generated from the network output and also has methods to manipulate binvox files in python ([reference](https://github.com/dimatura/binvox-rw-py)).
- create_dataset.py : Filters data from the entire `ShapeNet` dataset according to the arguments provided. This uses the metadata in the `ShapeNet` dataset to filter the models according to the requirements.
- occ_data.py : Main file which consists of the occupancy network, the dataloader and the training loop.
- viewvox : Used for the visulaizion of binvox file. ([reference](https://www.patrickmin.com/viewvox/))
