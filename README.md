# photonion
Implicit Likelihood Inference package trained on the SPHINX simulation to infer the escaped ionising luminosity of high-z galaxies 

## TODO LIST
- [ ] add example usage for how to load a model
- [ ] add example usage for creating a data array (use Carniani+24 galaxies) and computing $\dot{N}_{ion}$
- [ ] add model from Choustikov+24
- [ ] decide whether or not to add documentation about $f_{\rm esc}$ model
- [ ] think of anything else
- [ ] Richard: Proof read code

## Example Usage
Here's an easy example of how to use the package to infer $\dot{N}_{\rm ion}$ from photometry in JADES bands:
```python
# Example for GN-z11 (Bunker+23, Tacchella+23):
# Data: ["F090W", "F115W", "F150W", "F200W", "F277W", "F335M", "F356W", "F410M", "F444W", "z"]

import photonion # import package

SBIRegressor = photonion.SBIRegressor.from_config("../models", "SBI_JADES_nion") # load model
GN_z11 = np.array([-2.9, 1.2, 115.9, 144.4, 121.7, 132.9, 123.5, 114.9, 133.8, 10.6]).T # get data

data = photonion.convert_observational_data(data_vector) # convert data into useable features (Choustikov+24)
Nion = SBIRegressor.sample_summarized(data, n_samples=3000) # run pipeline, sample the posterior and summarize
print(f'GN-z11: {Nion[0][0]:.3f}+{(Nion[0][1]-Nion[0][0]):.3f}-{(Nion[0][0]-Nion[0][2]):.3f}') # print data
```
Which would return:
```
\dot{N}_{ion} Prediction for GN-z11: 53.081+0.393-0.460
```

## Citation
If you use this package for anything cool, a citation of the original paper should be included:
````bibtex
@ARTICLE{2024arXiv240509720C,
       author = {{Choustikov}, Nicholas and {Stiskalek}, Richard and {Saxena}, Aayush and {Katz}, Harley and {Devriendt}, Julien and {Slyz}, Adrianne},
        title = "{Inferring the Ionizing Photon Contributions of High-Redshift Galaxies to Reionization with JWST NIRCam Photometry}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Astrophysics of Galaxies, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2024,
        month = may,
          eid = {arXiv:2405.09720},
        pages = {arXiv:2405.09720},
          doi = {10.48550/arXiv.2405.09720},
archivePrefix = {arXiv},
       eprint = {2405.09720},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240509720C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
````

## Data
- SPHINX Data Release paper: https://ui.adsabs.harvard.edu/abs/2023OJAp....6E..44K/abstract
- SPHINX data: https://github.com/HarleyKatz/SPHINX-20-data
