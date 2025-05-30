# **VaporFit: Software for Automated Atmospheric Correction in FTIR Spectroscopy**

## **Description:**

VaporFit is an open-source software tool designed for automated atmospheric correction in FTIR spectroscopy.  
The program uses a least squares fitting method, dynamically adjusting subtraction coefficients to improve correction accuracy.  https://github.com/piobruzdpg/VaporFit/blob/main/README.md
VaporFit is available in compiled versions for macOS (M4) and Windows 11.

## **Authors:**

Przemysław Pastwa and Piotr Bruździak  
Department of Physical Chemistry, Gdańsk University of Technology  
Narutowicza 11-12, 80-233 Gdańsk, Poland

You are welcome to send us any comments or suggestions at *piotr.bruzdziak@pg.edu.pl*.

## **Package Contents:**

- Python 3.11 source file
- Compiled versions of the software for macOS (M4) and Windows 11 (both built with Nuitka; may be flagged as unsafe by Windows 11)  
  - If you encounter issues with the compiled version, run the Python script in your favorite Python IDE (NumPy, SciPy, and Matplotlib are required)
- Three sample data sets:  
  - **test_data_betaine_set.zip**: Contains 29 spectra of glycine betaine solutions and 2 atmospheric spectra  
  - **test_data_D2O-H2O_set.zip**: Contains 11 spectra of D₂O/H₂O mixtures and 3 atmospheric spectra
  - **test_data_urea_evaporation_set.zip**: Contains 20 spectra of evaporating urea solution and 2 atmospheric spectra
- User Guide

## **Features:**

- **Spectral Files**: Supports CSV and DPT formats.
- **Correction Algorithm**: Automatic optimization of subtraction coefficients based on least squares fitting.
- **Optional Tools**: "Optimum Search" tool for analyzing optimal SG parameters, PCA analysis, and checking correction coefficients.
- **Saving Results**: Exporting reports and corrected spectra in CSV format.

## **References:**

If you use VaporFit in your research, you must cite the following publications:

- Pastwa P., Bruzdziak P., *VaporFit: An Open-Source Software for Accurate Atmospheric Correction of FTIR Spectra*,
  *Physical Chemistry Chemical Physics*, 2025, 
  doi:[10.1039/D5CP01007A](https://doi.org/10.1039/D5CP01007A)

- Bruzdziak P., *Vapor correction of FTIR spectra – A simple automatic least squares approach*,  
  *Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy*, 2019, **223**, 1–4,  
  doi:[10.1016/j.saa.2019.117373](https://doi.org/10.1016/j.saa.2019.117373)
