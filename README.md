
# Off-TANet:A lightweight neural micro-expression recognizer with optical flow features and integrated attention mechanism

### How to Run the Code
1.Install packages mentioned in **requirements.txt** 

`pip install -r requirements.txt` 

2.Modify arguments in **train_arg.py**  

3.Get CASME,CASME2 and CASME-2 datasets from the link below,put the cropped pictures under the **dataset** directory. 

The name of the subfolders should be **casme1_cropped**,**casme2_cropped** and **casme^2_cropped** 

CASME - http://fu.psych.ac.cn/CASME/casme.php 

CASME2 - http://fu.psych.ac.cn/CASME/casme2.php 

CASME-2 - http://fu.psych.ac.cn/CASME/cas(me)2.php 

4.Run the code 

`python train.py` 

The results can be seen in this chart below.

| Model | UAR | UF1 |Total Params | Total Flops | Total MemR+W |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Off-ApexNet | 0.5832 | 0.5650 | 2.66M | 3.87M | 10.35MB |
| STSTNet | 0.5584 | 0.5399 | 162,051 | 526.98K | 0.78MB |
| Dual-Inception | 0.6167 | 0.5814 | 6.45M | 12.64M | 26.27MB |
| MACNN | 0.6835 | 0.6660 | 70.57M | 793.67M | 297.86MB |
| Micro-Attention | 0.7086 | 0.7021 | 53.38M | 1.0G | 237.97MB |
| Off-TANet | 0.7315 | 0.7242 | 59,403 | 30.08M | 5.64MB |

The required Python packages are in **requirements.txt**, and other environments of ours are as follows: 

Operating system: **Ubuntu 16.04.6 LTS**

CPU: **Intel(R) Xeon(R) Gold 5118 CPU@ 2.30GHz**

GPU: **Tesla K80 (10G video RAM)**

CUDA Version: **9.0**