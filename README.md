This repository contains relevant information of our research article [*Pre-trained protein language model sheds new light on the prediction of Arabidopsis protein–protein interactions*](https://link.springer.com/article/10.1186/s13007-023-01119-6).

### Reproduce results quickly
We provide a file of [conda virtual environment](https://github.com/keiwo/ESM_Ara_PPIs/blob/main/environment.yml) for you to refer via `conda env create -f environment.yml`. After the Python environment is ready, you can run the script `python MLP.py` in terminal and it use GPU default.

### Use data yourself
#### Data format
For training and testing datasets, you need to have protein pair and their label separated by tab character like `protein1 protein2 label`. Please note that ESM only received protein sequences less than 1022 when we used before, and things may have changed right now.
#### Sequence embedding
Please follow [this instruction](https://github.com/facebookresearch/esm) to extracts embeddings from ESM. After that, you should extract embedding from raw result and integrate all embeddings into Python dictionary like `{"protein_name": torch.tensor()}`. Then save it as `.pkl` file.
### Train and test
Replace the corresponding file names in `MLP.py` with yours and you will get PR value of testing dataset after running the script.
### Acknowledgment
We express our thanks to these researches.

>Alley EC, Khimulya G, Biswas S, AlQuraishi M, Church GM. Unified rational protein engineering with sequence-based deep representation learning. Nat Methods. 2019 Dec;16(12):1315-1322.
>
>Chen M, Ju CJ, Zhou G, Chen X, Zhang T, Chang KW, Zaniolo C, Wang W. Multifaceted protein-protein interaction prediction based on Siamese residual RCNN. Bioinformatics. 2019 Jul 15;35(14):i305-i314.
>
>Elnaggar A, Heinzinger M, Dallago C, Rehawi G, Wang Y, Jones L, Gibbs T, Feher T, Angerer C, Steinegger M, Bhowmik D, Rost B. ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. IEEE Trans Pattern Anal Mach Intell. 2022 Oct;44(10):7112-7127.
>
>Rao R, Bhattacharya N, Thomas N, Duan Y, Chen X, Canny J, Abbeel P, Song YS. Evaluating Protein Transfer Learning with TAPE. Adv Neural Inf Process Syst. 2019 Dec;32:9689-9701.
>
>Rives A, Meier J, Sercu T, Goyal S, Lin Z, Liu J, Guo D, Ott M, Zitnick CL, Ma J, Fergus R. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proc Natl Acad Sci U S A. 2021 Apr 13;118(15):e2016239118.
>
>Sledzieski S, Singh R, Cowen L, Berger B. D-SCRIPT translates genome to phenome with sequence-based, structure-aware, genome-scale predictions of protein-protein interactions. Cell Syst. 2021 Oct 20;12(10):969-982.e6.
>
>Song B, Luo X, Luo X, Liu Y, Niu Z, Zeng X. Learning spatial structures of proteins improves protein-protein interaction prediction. Brief Bioinform. 2022 Mar 10;23(2):bbab558.
>
>Szymborski J, Emad A. RAPPPID: towards generalizable protein interaction prediction with AWD-LSTM twin networks. Bioinformatics. 2022 Aug 10;38(16):3958-3967.
>
>Zhao J, Lei Y, Hong J, Zheng C, Zhang L. AraPPINet: An Updated Interactome for the Analysis of Hormone Signaling Crosstalk in *Arabidopsis thaliana*. Front Plant Sci. 2019 Jul 5;10:870.
>
>Zheng J, Yang X, Huang Y, Yang S, Wuchty S, Zhang Z. Deep learning-assisted prediction of protein-protein interactions in *Arabidopsis thaliana*. Plant J. 2023 May;114(4):984-994. 










