# Installation example

*Written by Jin Hyun Cheong*

Open the current notebook in Google Colab and run the cell below to install Py-Feat. Make sure to `Restart Runtime` so that the installation is recognized. 

# Install Py-Feat from source.
!git clone https://github.com/cosanlab/feat.git`  
!cd feat && pip install -q -r requirements.txt
!cd feat && pip install -q -e . 
!cd feat && python bin/download_models.py
# Click Runtime from top menu and Restart Runtime! 

Make sure you Restart Runtime before this next step.

# Check Fex class installation
from feat import Fex
fex = Fex()

# Check Detector class installation
from feat import Detector
detector = Detector()