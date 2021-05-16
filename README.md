# IPNV2_pytorch
This is an pytorch implementation of "IPN-V2 and OCTA-500: Methodology and Database for Retinal Image Segmentation". 
# Dataset
The dataset OCTA500 is available at: https://ieee-dataport.org/open-access/octa-500.

Use this dataset, you need to preprocess the downloaded labels. Two operations are required：

1）Rotate 90 degrees：To align with 3D data.

2）Change the gray value of the label image to：0-background,1-FAZ,(2-RV，if you need).

# Related Papers:
-Mingchao Li, Yerui Chen, Zexuan Ji, Keren Xie, Songtao Yuan, Qiang Chen, and Shuo Li.“Image projection network: 3D to 2D image segmentation in OCTA images,” IEEE Trans. Med. Imaging, vol. 39, no. 11 pp. 3343-3354, 2020.

-Mingchao Li, Yuhan Zhang, Zexuan Ji, Keren Xie, Songtao Yuan, Qinghuai Liu and Qiang Chen. "IPN-V2 and OCTA-500: Methodology and Dataset for Retinal Image Segmentation," arXiv:2012.07261.
# Related Codes:
IPN: https://github.com/chaosallen/IPN_tensorflow.

IPN-V2: https://github.com/chaosallen/IPNV2_pytorch.

# Network Structure
![image](./IPNV2.jpg)
