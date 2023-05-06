# diss

Here is the code for my final year project

The Joey folder is for T2G model, it contains the best model discussed in the Paper however no data due to size, you can try to download the PHOENIX14T dataset online here: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/

The other fold is a fixed up version of Ben Saunders Progressive Transformer, it acts in the same described in his readme so I left it in. You can download the 2 smallest models discussed for T2P and T2G2P and the full dataset from my google drive: 
Dataset: https://drive.google.com/file/d/11jrpbTVnSMU0_4_3GGZwU6mipWA93moQ/view?usp=share_link
T2P: https://drive.google.com/file/d/10mLUPAI1qWS12MtyFkLg2gunHrFmWxLa/view?usp=share_link
T2G2P: https://drive.google.com/file/d/1NLrsMR0ZZ945bDciBAQPvgIsmtJYWbcS/view?usp=share_link

Put the data in a folder called sign_data to match the configs or change the config file path. The glossv2 is the predicted gloss from the T2G only for when doing evaluation but you'll have to change the source text to gloss v2 in the config before running the test.
