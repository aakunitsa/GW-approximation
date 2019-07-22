# GW-approximation
Reference Python implementation of GW 

**Dependencies**:
- Python 3 (version >= 3.6)
- Psi4 1.2 (please refer to http://vergil.chemistry.gatech.edu/nu-psicode/install-v1.2.html for installation instrunctions)

**Brief description of the contents**:
- GW_HF.ipynb contains a simple vanilla implementation of analytic GW@HF with a detailed review of theory and some derivations.
- RI-GW.ipynb implements RI approximation as well as SPA
- GW.py is a rather full-featured implementation of the GW class with the following capabilities:
   1. Resolution of identity
   2. Contour deformation 
   3. RPA for W
   4. evGW 
- Folder tests contains some sample calcuations

**Author**:

Alexander A. Kunitsa (UIUC)
