# Efficient Document Shadow Removal with Contrast-Aware Guidance


## <a name="todo"></a>:climbing:TODO
- [ ] Support Multi-gpu inference.
- [x] Clean up the code and release the code along with the checkpoint.
- [x] This repo is created.

## <a name="quick_start"></a>:flight_departure:Quick Start
###  Clone the repository
```
git clone https://github.com/yifan613/Shadow-Contrast.git
cd your-repo
```
##  Create and activate the environment
```
conda create -n shadowcontrast python=3.8 -y
conda activate shadowcontrast
pip install -r requirements.txt
```
##  Run the project
Please make sure to update the paths in `conf.yml` according to your local setup.  
The pretrained model used in this project can be downloaded from the [Releases](https://github.com/yifan613/Shadow-Contrast/releases) page.  
Test datasets and the corresponding inference results can be found at the following OneDrive link:  
👉 [Googledrive link](https://drive.google.com/drive/folders/13WhAj11Y9Bc-vQJZkRwrLdcb-Boako1v?usp=drive_link) 
```
python main.py
```
## Acknowledgement
This project is based on [DocDiff](https://github.com/Royalvice/DocDiff), we would like to thank the authors for their excellent work.
