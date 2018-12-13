# A Richly Annotated Pedestrian Dataset for Person Retrieval in Real Surveillance Scenarios

## Preparation
<font face="Times New Roman" size=4>

**Prerequisite: Caffe, Python 2.7 and Matlab.**

1. Install [Caffe](https://github.com/dangweili/caffe), Python 2.7 and Maltab.

2. Download and prepare the dataset as follow:
   
    RAP [Links](https://drive.google.com/open?id=1hoPIB5NJKf3YGMvLFZnIYG5JDcZTxHph)
    ```
    ./data/RAP_annotation/RAP_annotation.mat
    ./data/RAP_dataset/*.png
    ```

    if you want to use body parts for attribute recognition, please exec this command to generate body part images.
    ```
    cd data
    matlab -nodisplay -r 'rap2_part_extraction'
    ```
3. Download the imagenet pretrained models.
   ImageNet pretrained models which is used in finetuning [Baiduyun](https://pan.baidu.com/s/1IxZ6GrAFSfhT9Ipa7a-Zuw) or [GoogleDrive](https://drive.google.com/open?id=14p0MLyAdsoGaqfSGvj-IH59Ifo2Ojjqp).

</font>

## Pedestrian Attribute Recognition
<font face="Times New Roman" size=4>

1. SVM-based models
    a. feature extraction, including ELF and pretrained CNN features:
    ```
    cd features/ELF-v2.0-Descriptor
    matlab -nodisplay -r 'Feature_Extraction_elf'
    matlab -nodisplay -r 'Feature_PCA_elf'
    ```
    download pretrained CNN models and run the follow commands to extract cnn features.
    ```
    cd features/CNN-v1.0-Descriptor
    matlab -nodisplay -r 'imagenet_feature_extraction_caffenet_single'
    matlab -nodisplay -r 'imagenet_feature_extraction_resnet_single'
    matlab -nodisplay -r 'imagenet_feature_extraction_caffenet_parts' [optional]
    matlab -nodisplay -r 'imagenet_feature_extraction_resnet_parts' [optional]

    ```
    compile the liblinear
    ```
    cd person-attribute/utils/liblinear-master/matlab
    matlab -nodisplay -r 'make'
    ```

    b. use svm to train attribute classifiers 
    training: mixture clean and occlusion data, test: mixture clean and occlusion data.
    ```
    cd person-attribute/baseline-svm
    matlab -nodisplay -r 'v1_fullbody_analysis_mm' [one types of features]
    matlab -nodisplay -r 'v1_fullbody_analysis_mm_test' [all types of features]
    matlab -nodisplay -r 'v1_fullbody_analysis_mm_statistic' [summary the results]
    ```
    c. analysis of viewpoint
    training: maxture clean and occlusion data, test: mixture clean and occlusion data.
    ```
    cd person-attribute/baseline-svm
    matlab -nodisplay -r 'v2_fullbody_analysis_mm_viewpoint'
    matlab -nodisplay -r 'v2_fullbody_analysis_mm_viewpoint_statistic'

    ```
    training: clean, test: clean
    ```
    cd person-attribute/baseline-svm
    matlab -nodisplay -r 'v2_fullbody_analysis_cc'
    matlab -nodisplay -r 'v2_fullbody_analysis_cc_viewpoint'
    
    ```
    d. analysis of occlusion
    occlusion positions and types: training: clean, test: occlusion
    ```
    cd person-attribute/baseline-svm
    matlab -nodisplay -r 'v3_fullbody_analysis_co_test'
    matlab -nodisplay -r 'v3_fullbody_analysis_co_test_personvspersons'

    ```
    e. analysis of body parts
    ```
    cd person-attribute/baseline-svm
    matlab -nodisplay -r 'v4_parts_analysis_cc'
    matlab -nodisplay -r 'v4_parts_analysis_cc_test'
    ```

2. CNN-based models
    a. prepare the data splits.
    ```
    cd person-attribute/static
    matlab -nodisplay -r 'prepare_data'
    matlab -nodisplay -r 'prepare_data_parts'
    matlab -nodisplay -r 'prepare_data_binary'
    ```
    b. train the deep attribute classifiers with deepmar based on CaffeNet.
    For deepmar, acn, and their single attribute versions, the operators are in similar format.
    ```
    cd person-attribute/baseline-deepmar
    sh train_caffenet.sh
    sh test_caffenet.sh
    ```

</font>


## Attribute-based Person Retrieval
<font face="Times New Roman" size=4>

The product of multiple attributes' prediction probability are used for person retrieval.

1. generate the attributes for attribute-based person retrieval.
    ```
    cd person-attribute/baseline-search
    matlab -nodisplay -r 'generate_multiquery_index'
    matlab -nodisplay -r 'generate_query_names'
    python generate_query_names.py
    ```
2. generate the attribute score based on the trained models
    ```
    cd person-attribute/baseline-search
    matlab -nodisplay -r 'generate_svm_score'
    python generate_cnn_score.py
    python generate_cnn_score_binary.py
    ```
3. evaluate the attribute-based person retrieval
    ```
    cd person-attribute/baseline-search
    matlab -nodisplay -r 'evaluate_multiquery_attributes'
    ```

</font>


## Person Re-identification
<font face="Times New Roman" size=4>

1. hand-crafted features/pretrained cnn features with L2/XQDA/KISSME
    a. feature extraction, incluidng ELF, LOMO, GOG (window), JSTL
    ```
    cd features/ReID_GOG_v1.01
    matlab -nodisplay -r 'Feature_Extraction_gog'
    cd features/CNN-v1.0-Descriptor
    matlab -nodisplay -r 'jstl_feature_extraction_single'
    cd features/LOMO_XQDA/code
    matlab -nodisplay -r 'Feature_Extraction_lomo'
    ```
    b. feature evaluation
    ```
    cd person-reid/evaluation
    matlab -nodisplay -r 'rap2_evaluation_features'
    ```
2. end-to-end feature learning
    
    a. generate the data split file for training.
    ```
    cd person-reid/static
    matlab -nodisplay -r 'generate_att_trainval_test'
    matlab -nodisplay -r 'generate_ide_trainval_test'
    matlab -nodisplay -r 'generate_ide_att_trainval_test'
    matlab -nodisplay -r 'generate_ide_att_trainvaltest' 
    ```

    b. training with only ID classification loss, such as CaffeNet, ResNet50/ResNet101/ResNet152, DenseNet121, MSCAN.
    ```
    cd person-reid/baseline-IDE
    sh train_caffenet.sh
    ```

    c. training with only attribute classification loss
    ```
    cd person-reid/baseline-att
    sh train_caffenet.sh
    sh test_caffenet.sh [optional for attribute classification]
    ```

    d. training with attribute and ID classification losses
    ```
    cd person-reid/baseline-IDE-att
    sh train_caffenet.sh
    sh test_caffenet.sh [optional for attribute classification]
    ```
    
    e. deep feature extraction and evaluation.
    ```
    cd person-reid/evaluation
    sh rap2_feature_extraction_resnet.sh [one model per time]
    matlab -nodisplay -r 'rap2_test' [one model per time]
    ```
3. cross-day person retrieval
    a. person retrieval in the same day as query or the different day as query.
    ```
    cd person-reid/evaluation
    matlab -nodisplay -r 'rap2_test_control_single_cross'
    ```
    b. person retrieval from different day as query. The appearance would be partially different for the same person across different days.
    ```
    cd person-reid/evaluation
    matlab -nodisplay -r 'rap2_test_control_single_cross_quantively'
    ```
4. identity-level attribute vs. instance-level attributes for person re-identification.
    a. generate identity-level attributes from instance-level attributes for training.
    ```
    cd person-reid/static
    matlab -nodisplay -r 'generate_ide_att_trainval_test_control'
    ```
    b. train: instance-level attributes. The default setup. 
    ```
    cd person-reid/baseline-IDE-att
    sh train_caffenet.sh
    ```
    c. train: identity-level attributes. 
    ```
    cd person-reid/baseline-IDE-att-control
    sh train_caffenet.sh
    ```
    d. feature extraction and evaluation.
    ```
    cd person-reid/evaluation
    sh rap2_reid_extraction_resnet_control_identity_instance.sh 
    matlab -nodisplay -r 'rap2_test_control_identity_identity'
    matlab -nodisplay -r 'rap2_test_control_identity_instance'
    matlab -nodisplay -r 'rap2_test_control_instance_identity'
    ```

</font>


## Citation
<font face="Times New Roman" size=4>
Please cite this paper in your publications if it helps your research:
</font>

```
@article{li2018richly,
    title={A Richly Annotated Pedestrian Dataset for Person Retrieval in Real Surveillance Scenarios},
    author={Li, Dangwei and Zhang, Zhang and Chen, Xiaotang and Huang, Kaiqi},
    journal={IEEE Transactions on Image Processing},
    volume={28},
    number={4},
    pages={1575--1590},
    year={2019},
    publisher={IEEE}
}
```
