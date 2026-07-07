+ **Our code** is in the folder of "RCG-Net". We have uploaded the code, weight, and training sets now.  
+ **5 testing sets** and **1 testing set with varying noise types & levels** used in our paper are shown in the folder of "TestingSet"
+ **The output of our RCG-Net** is shown in the folder of "RCG-Net-output"  
+ **Codes and results of 14 SOTA UIE methods** are shown in the folder of "Comparison", some of their checkpoints are too large so we didn't upload them, but you can get them from the link in their paper.
+ Results of all methods under varying noise types & levels are shown in the folder of "Comparison_under_noise"
+ **The codes of all IQA metrics** used in our paper **(PSNR & SSIM & UIQM & UCIQE & NIQE & URanker)** are given in the folder of "IQA_Evaluation". The subfolder "OtherNotUsedInPaper" also includes some metrics we didn't use like LPIPS & NUIQ.
+ **The results of Ablation Study** including the structure, input and loss are in the folder of "Ablation Study"

---

# The subjective results of all 14 methods
We compared RCG-Net with 14 UIE methods to demonstrate the superiority of our approach. These methods include two restoration method based on physical models: **GDCP (2018)** and **Retinex (2014)**; two traditional methods based on non-physical models: **Fusion (2012)** and **ZSRM (2024)**; ten data-driven and hybrid methods: **FUnIE-GAN (2020)** (GAN-based), **CCMSR-Net (2023)** (Multi-scale Retinex + CNN + Transformer) denoted as **CCMSR**
in this paper, **PUGAN (2023)** (a physical model guided GAN), **WaterNet (2019)** (CNN-based), **Ucolor (2021)** (CNN + GDCP), **DICAM (2024)** (CNN based), **U-Shape Transformer (2023)** (Transformer-based), **CTM (2024)** (Transformer + CNN), **UWMamba (2024)** (Mamba + CNN), and **GuidedHybSensUIR (2025)** (Color Balance Prior + CNN + Transformer) (which we denote as HybSens). 

<u>Due to space constraints and their relatively poor performance, **GDCP**, **Fusion**, **ZSRM**, and **FUnIE** were excluded from the quantitative and qualitative comparisons in our paper.</u> **But we have shown the results of all 14 methods below.** (You can also seen the complete results in the folder of "Comparison"~)

## 1. Evaluation in Common and Challenging Degradation Scenarios (Test-110 & Test-C60 & Test-U45)
#### The subjective results of 4 methods that didn't present in the paper:
<img src="./README_Img/1-NoCompared4.png" alt="relative poor methods" title="Evaluation in Common and Challenging Degradation Scenarios">

---

#### The subjective and objective results of 10 better methods that have presented in the paper:
<img src="./README_Img/1_Common_and_Challenging.jpg" alt="paperUsed" title="Evaluation in Common and Challenging Degradation Scenarios">

---

**Quantitative comparison among different UIE methods on FR testing set: Test-110**

| Method   | PSNR↑   | SSIM↑   | UIQM↑   | UCIQE↑  | NIQE↓   | URanker↑ |
|----------|---------|---------|---------|---------|---------|----------|
| Retinex  |  17.312  |   0.491    |**<u>0.905</u>**|   0.582    |   42.981   | **<u>2.423</u>** |
| CCMSR  |  22.699  |   0.794    | 0.732 |   0.582    |   44.318   |  1.958 |
| PUGAN    |**25.678**|<u>0.861</u>|  <u>0.790</u>  |<u>0.621</u>| **41.368** |   <u>2.205</u>   |
| WaterNet | 23.106  | 0.822   | 0.722   | 0.590   | 42.916  | 1.748    |
| Ucolor   | 23.686  | 0.828   | 0.725   | 0.589   | 44.491  | 1.898    |
| DICAM    | <u>24.815</u> | 0.849   | 0.764   | 0.611   | 43.926  | 2.072    |
| UShape   | 21.855  | 0.774   | 0.627   | 0.559   | 43.859  | 1.592    |
| CTM      | 23.247  | 0.829   | 0.689   | 0.605   | 46.077  | 2.005    |
| UWMamba  | **<u>26.145</u>** | **<u>0.878</u>** | 0.775   | **0.627** |**<u>41.007</u>**| 2.185    |
| HybSens  | 24.354  | **0.871** | 0.751   | 0.609   | 41.715  | 2.077    |
| RCG-Net  | 22.968  | 0.827   | **0.828** | **<u>0.630</u>** | <u>41.641</u>  | **2.295** |


**Quantitative comparison among different UIE methods on NR testing set: Test-C60 and Test-U45**

| Method   | UIQM↑ (C60) | UIQM↑ (U45) | UCIQE↑ (C60) | UCIQE↑ (U45) | NIQE↓ (C60) | NIQE↓ (U45) | URanker↑ (C60) |   URanker↑ (U45) |
|----------|--------------|------------|---------------|-------------|---------------|-------------|-----------------|----------------|
| Retinex  |**<u>0.809</u>**|**<u>0.950</u>**| 0.546   | 0.574       | 45.683        | 42.712      | **<u>2.122</u>**|**<u>2.629</u>**|
| CCMSR  | 0.437 | 0.768 | 0.514   | 0.572       | 46.475        | 44.026      | 1.127 | 2.152 |
| PUGAN    | 0.573       | 0.799      |**<u>0.574</u>**|**<u>0.605</u>** | 43.550        | 43.342      | **1.868**       | <u>2.466</u>   |
| WaterNet | 0.552        | 0.747      | 0.533         | 0.572       | 41.438        | 42.813      | 1.262           | 1.861          |
| Ucolor   | 0.412        | 0.788      | 0.500         | 0.573       | 47.765        | 43.521      | 1.080           | 2.152          |
| DICAM    | 0.541        | 0.794      | 0.544         | 0.579       | 45.925        | 41.882      | 1.536           | 2.022          |
| UShape   | 0.420        | 0.656      | 0.497         | 0.534       | 43.328        | 47.182      | 1.047           | 1.603          |
| CTM      | 0.387        | 0.795      | 0.528         | 0.593       | 48.041        |<u>41.439</u>| 1.351           | 2.185          |
| UWMamba  | 0.551        |<u>0.808</u>| <u>0.551</u>  | <u>0.601</u>|**<u>38.320</u>**| **41.189**| 1.558           | 2.390          |
| HybSens  | <u>0.574</u> | 0.796      | 0.542         | 0.589       | <u>40.498</u> |**<u>39.847</u>**| 1.486       | 2.328          |
| RCG-Net  | **0.644**    | **0.853**  | **0.570**     | **0.602**   | **39.689**    | 42.656      | <u>1.810</u>    |**2.497**       |

PS: CCMSR, WaterNet, CTM, HybSens and our RCG-Net were trained on UIEB-780, while the others employed their publicly available pretrained versions. The top three metric scores are represented from best to worst as **<u>bold and underlined</u>**, **bold**, and <u>underlined</u>, respectively.


## 2. Evaluation in Lighting Degradation Scenarios (Test-NUID & Test-OD)
#### The subjective results of 4 methods that didn't present in the paper:
<img src="./README_Img/2-NoCompared4.png" alt="relative poor methods" title="Evaluation in Lighting Degradation Scenarios">

---

#### The subjective and objective results of 10 better methods that have presented in the paper:
<img src="./README_Img/2_Lighting_Degradation.jpg" alt="paperUsed" title="Evaluation in Lighting Degradation Scenarios">

---

**Quantitative comparison among different UIE methods on NR testing set: Test-NUID and Test-OD**

| Method   | UIQM↑ (NUID) | UIQM↑ (OD) | UCIQE↑ (NUID) | UCIQE↑ (OD) | NIQE↓ (NUID) | NIQE↓ (OD) | URanker↑ (NUID) |   URanker↑ (OD) |
|----------|--------------|------------|---------------|-------------|---------------|-------------|-----------------|----------------|
| Retinex  |**<u>0.789</u>**|**<u>0.624</u>**| 0.514   | 0.492       | 46.286        | 52.954      | **<u>1.953</u>**| 1.280          |
| CCMSR  | 0.500 | 0.328 | 0.535   | 0.511       | 46.058        | 44.215      | 1.339 |  0.914          |
| PUGAN    | 0.594        |<u>0.542</u>| 0.561         | 0.539       | 41.670        | 44.601      | 1.653           | <u>1.317</u>   |
| WaterNet | 0.569        | 0.455      | 0.533         | 0.512       | **40.366**    |**<u>35.281</u>**| 1.256       | 0.826          |
| Ucolor   | 0.335        | 0.435      | 0.529         | 0.540       | 47.248        | 47.948      | 1.301           | 1.177          |
| DICAM    | 0.582        | 0.360      | 0.553         | 0.544       | 45.312        | 40.176      | 1.578           | 1.122          |
| UShape   | 0.273        | 0.499      | 0.486         | 0.502       | 43.077        | 42.343      | 0.964           | 0.777          |
| CTM      | 0.292        | 0.448      | 0.537         |**<u>0.569</u>**| 48.255     | 45.012      | 1.354           | 1.261          |
| UWMamba  | 0.547        | 0.473      | <u>0.564</u>  | <u>0.560</u>|**<u>39.639</u>**| **35.532**| 1.635           | 1.150          |
| HybSens  | **0.656**    | **0.587**  | **0.571**     | **0.561**   | <u>40.467</u> |<u>40.003</u>| <u>1.710</u>    | **1.353**      |
| RCG-Net  | <u>0.596</u> | 0.541      | **<u>0.576</u>** | 0.555    | 41.173        | 41.115      | **1.891**       |**<u>1.413</u>**|

>PS: CCMSR, WaterNet, CTM, HybSens and our RCG-Net were trained on UIEB-780, while the others employed their publicly available pretrained versions. The top three metric scores are represented from best to worst as **<u>bold and underlined</u>**, **bold**, and <u>underlined</u>, respectively.

## 3. Evaluation of Generalization Ability
To further validate the model’s exceptional generalization ability and SOTA performance, we compared RCG-Net with ten other methods across all selected NR metrics (UIQM, UCIQE, NIQE, URanker) on five different datasets one by one. 

**The recording rules** were as follows: Before comparisons, all heat map positions were initialized to 0, indicating equal performance among all methods. Each method was then evaluated against the other nine methods across all four metrics on all five datasets, resulting in 20 pairwise comparisons per method. For each metric on a specific dataset, if method x outperformed method y, the performance score at position (x,y) was incremented by 1, while (y,x) was decremented by 1, as a result of which the scores were restricted to a range of [-20,20]. Following all comparisons and statistical analyses, the results were visualized as a heat map below:

<img src="./README_Img/6-HeatMap.jpg" alt="paperUsed" title="Evaluation of Generalization Ability">

>The performance comparison heatmap shows how each UIE method is superior to / inferior to other methods across four NR metrics on five test datasets. The value at position (x,y) represents the performance score of method x relative to method y, calculated as the number of wins minus the number of losses in their direct comparisons, with a range of [-20,20]. For instance, a value of 8 at (RCG-Net, UWMamba) indicates that RCG-Net won 14 out of 20 comparisons against UWMamba. The sum scores on the right side represents the overall performance of each method compared to others, with a range of [-180, 180]. The larger the better.



## 4. Evaluation on Color Restoration Accuracy (Color-Checker7)

#### The subjective results of 4 methods that didn't present in the paper:
<img src="./README_Img/3-NoCompared4.png" alt="relative poor methods" title="Evaluation on Color Restoration Accuracy">

---

#### The subjective and objective results of 10 better methods that have presented in the paper:
<img src="./README_Img/3_Color_Restoration_Accuracy.jpg" alt="paperUsed" title="Evaluation on Color Restoration Accuracy">

---

**Quantitative comparison of the average CIEDE2000 score among different UIE methods on Color-checker7**

| **Method**   | GDCP  | Retinex | ZSRM  | Fusion | CCMSR | PUGAN | FUnIE | WaterNet |
|----------|-------|---------|-------|--------|-------|-------|----------|--------|
| **AVG**  | 16.22 | 14.50   | 14.59 | 11.58 | 12.69 | 11.85 | 14.40 | 11.82    |

| **Method**   | Ucolor | DICAM | UShape | CTM   | UWMamba | HybSens | Ours |
|----------|-------|--------|-------|-------|---------|---------|---------|
| **AVG** | 11.78  | 11.49 | **10.94** | 11.91 | <u>11.40</u> | 11.96 | **<u>10.89</u>** |

>PS: The top three metric scores are represented from best to worst as **<u>bold and underlined</u>**, **bold**, and <u>underlined</u>, respectively.


