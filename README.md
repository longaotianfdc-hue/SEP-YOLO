# SEP-YOLO
**Fengming Zhang，Tao Yan\*，Jianchao Huang **

> Abstract—Transparent object instance segmentation presents significant challenges in computer vision, due to the inherent properties of transparent objects, including boundary blur, low
contrast, and high dependence on background context. Exist
ing methods often fail as they depend on strong appearance
cues and clear boundaries. To address these limitations, we
propose SEP-YOLO, a novel framework that integrates a dual
domain collaborative mechanism for transparent object instance
segmentation. Our method incorporates a Frequency Domain
Detail Enhancement Module, which separates and enhances weak
high-frequency boundary components via learnable multi-scale
filtering. We further design a multi-scale spatial refinement
stream, which consists of a Content-Aware Alignment Neck and
a Multi-scale Gated Refinement Block, to ensure precise feature
alignment and boundary localization in deep semantic features.
We also provide high-quality instance-level annotations for the
Trans10K dataset, filling the critical data gap in transparent
object instance segmentation. Extensive experiments on the
Trans10K and GVD datasets show that SEP-YOLO achieves
state-of-the-art (SOTA) performance.

## Network Architecture
![architecture](images/architecture.png)


## Quantitative Comparisons
![QuantitativeComparisons](images/QuantitativeComparisons.png)

## APs Comparisons

![APsComparisons](images/APsComparisons.png)
