# MMCFusion
### Infrared and Visible Image Fusion Based on Multi-modal and Multi-scale Cross-compensation

Meitian Li, Jing Sun, Heng Ma, Fasheng Wang and Fuming Sun

***************
### Abstract : 

To tackle the challenge of detail information loss and redundancy in existing infrared and visible
 fusion algorithms, this paper proposes a novel infrared and visible image fusion network based on
 Multi-modal and Multi-scale Cross-compensation (MMCFusion).The proposed network incorporates
 an Upper-Lower Level Cross-Compensation Module (ULCC) that integrates features from adjacent
 levels to enhance the richness and diversity of feature representations. Additionally, the feature Differ
ence Cross-Compensation Rule (DCCR)isintroducedtofacilitate cross-compensation of upper-lower
 level information through a differential approach.This design enhances the complementarity between
 features and effectively mitigates the problem of detail information loss prevalent in conventional
 methods. To further augment the model’s ability to detect and represent objects across various scales,
 wealsodevise the Multi-Scale Cascaded (MSC)modulethateffectivelyintegrates feature information
 frommultiple scales, thereby improving the model’sadaptability to diverse objects. Finally, we design
 a Texture Enhancement Module (TEM) to capture and retain local structures and texture information
 in the image, thereby providing richer detail representation after processing. To comprehensively
 capture multi-modal information and perform remote modeling, we employ PVT v2 to construct a
 dual-stream Transformer encoder, which can capture valuable information at multiple scales and pro
vide robust global modeling capabilities, thereby improving the fusion results. The efficacy of the
 proposed method is rigorously evaluated on several datasets, including infrared and visible datasets
 such as MSRS, TNO, and RoadScene, and medical imaging dataset like PET-MRI. Experimental re
sults demonstrate that the proposedmethodsignificantlyoutperformsexistingstate-of-the-artmethods
 in both visual quality and quantitative metrics, confirming its generalization capability across various
 datasets. 

 All pre-trained models can be downloaded at XXXXXXXXXX

 ## Quick Run

```python
python train.py
```

## Test (Evaluation)

```python
python test.py
```

 
