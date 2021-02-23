# UHI-calculation
This repository contains the code to calculate the Urban Heat Island (UHI) from gridded data as described in

**Valmassoi and Keller (2021), Representing the Urban Heat Island in Gridded Datasets, Advances in Science and Research.
DOI: TBD**


**How to use**:
Modify the _main.py_ accordingly to your case study. Fix the input variables so that they are all 1-D.
The function that calculates the UHI is **calc_uhi**, which is contained in _uhi_calculation.py_

Two examples are given:
1. the basic usage with a single timestep of the temperature field.
2. A possible way to treat time-series calculations.

The repository includes the main dependencies in "requirements.txt".

**Additional features**:
A simple plotting routine with Basemap: **quick_uhi_plot**. The _main.py_ gives an example on how to use it.



This program is a free software distributed under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 (GNU-GPLv3). You can redistribute and/or modify by citing the mentioned 
publication, but WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.