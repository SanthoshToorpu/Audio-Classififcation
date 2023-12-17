# Audio Classification with ESC-50 Dataset

```markdown

## 1. Dataset

The project utilizes the ESC-50 dataset, a collection of 2000 environmental audio recordings equally distributed across 50 classes. Each class represents a different sound event, making it suitable for audio classification tasks. You can find the dataset [here]([link_to_dataset](https://github.com/karolpiczak/ESC-50)).

## 2. Source

The source code for this project can be found in the [src](src) directory.

## 3. Technologies Used

- Python
- Librosa (for audio processing)
- TensorFlow (or any other machine learning framework of your choice)
- Matplotlib (for plotting results)

Make sure to install the required dependencies using:

```bash
pip install -r requirements.txt
```

## 4. Final Result

The model achieved impressive results in audio classification, with accuracy reaching [accuracy_percentage]% on the validation set. For a detailed overview of the training process and performance metrics, refer to the notebook.

### How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/SanthoshToorpu/Audio-Classififcation/tree/main
   ```

2. Install dependencies:

   ```bash
   cd your-repo
   pip install -r requirements.txt
   ```

3. Run the audio classification script:

   ```bash
   python src/audio_classification.py
   ```

   This script will load the trained model and perform audio classification on a sample audio file.
   ## Model Statistics
   ## Model - 1
   Model: "sequential_9"
   ```python
   _________________________________________________________________
    Layer (type)                Output Shape              Param #   
   =================================================================
    conv2d_42 (Conv2D)          (None, 11, 85, 16)        160       
                                                                    
    conv2d_43 (Conv2D)          (None, 9, 83, 16)         2320      
                                                                    
    conv2d_44 (Conv2D)          (None, 7, 81, 32)         4640      
                                                                    
    conv2d_45 (Conv2D)          (None, 5, 79, 32)         9248      
                                                                    
    conv2d_46 (Conv2D)          (None, 3, 77, 64)         18496     
                                                                    
    conv2d_47 (Conv2D)          (None, 1, 75, 32)         18464     
                                                                    
    global_average_pooling2d_3  (None, 32)                0         
     (GlobalAveragePooling2D)                                       
                                                                    
    dense_12 (Dense)            (None, 32)                1056      
                                                                    
    dense_13 (Dense)            (None, 10)                330       
                                                                    
   =================================================================
   Total params: 54714 (213.73 KB)
   Trainable params: 54714 (213.73 KB)
   Non-trainable params: 0 (0.00 Byte)
   _________________________________________________________________
   ```
   - **Final Training Loss:** 0.07247831672430038
   - **Final Validation Loss:** 0.7191997766494751
   - **Final Training Accuracy:** 0.96875
   - **Final Validation Accuracy:** 0.8583333492279053
   - **Highest Training Accuracy:** 0.971875011920929
   - **Highest Validation Accuracy:** 0.8916666507720947

   ![mode1](https://github.com/SanthoshToorpu/Audio-Classififcation/assets/90833739/117ef1c8-1936-4983-9186-025efc00d468)
   ![mh](https://github.com/SanthoshToorpu/Audio-Classififcation/assets/90833739/0fac0d75-3316-4769-a88c-a2ff0d970468)

   ## Model - 2
   Model: "efficientnetv2-b0"
```python   
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_6 (InputLayer)        [(None, None, None, 3)]      0         []                            
                                                                                                  
 rescaling_5 (Rescaling)     (None, None, None, 3)        0         ['input_6[0][0]']             
                                                                                                  
 normalization_5 (Normaliza  (None, None, None, 3)        0         ['rescaling_5[0][0]']         
 tion)                                                                                            
                                                                                                  
 stem_conv (Conv2D)          (None, None, None, 32)       864       ['normalization_5[0][0]']     
                                                                                                  
 stem_bn (BatchNormalizatio  (None, None, None, 32)       128       ['stem_conv[0][0]']           
 n)                                                                                               
                                                                                                  
 stem_activation (Activatio  (None, None, None, 32)       0         ['stem_bn[0][0]']             
 n)                                                                                               
                                                                                                  
 block1a_project_conv (Conv  (None, None, None, 16)       4608      ['stem_activation[0][0]']     
 2D)                                                                                              
                                                                                                  
 block1a_project_bn (BatchN  (None, None, None, 16)       64        ['block1a_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block1a_project_activation  (None, None, None, 16)       0         ['block1a_project_bn[0][0]']  
  (Activation)                                                                                    
                                                                                                  
 block2a_expand_conv (Conv2  (None, None, None, 64)       9216      ['block1a_project_activation[0
 D)                                                                 ][0]']                        
                                                                                                  
 block2a_expand_bn (BatchNo  (None, None, None, 64)       256       ['block2a_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block2a_expand_activation   (None, None, None, 64)       0         ['block2a_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block2a_project_conv (Conv  (None, None, None, 32)       2048      ['block2a_expand_activation[0]
 2D)                                                                [0]']                         
                                                                                                  
 block2a_project_bn (BatchN  (None, None, None, 32)       128       ['block2a_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block2b_expand_conv (Conv2  (None, None, None, 128)      36864     ['block2a_project_bn[0][0]']  
 D)                                                                                               
                                                                                                  
 block2b_expand_bn (BatchNo  (None, None, None, 128)      512       ['block2b_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block2b_expand_activation   (None, None, None, 128)      0         ['block2b_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block2b_project_conv (Conv  (None, None, None, 32)       4096      ['block2b_expand_activation[0]
 2D)                                                                [0]']                         
                                                                                                  
 block2b_project_bn (BatchN  (None, None, None, 32)       128       ['block2b_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block2b_drop (Dropout)      (None, None, None, 32)       0         ['block2b_project_bn[0][0]']  
                                                                                                  
 block2b_add (Add)           (None, None, None, 32)       0         ['block2b_drop[0][0]',        
                                                                     'block2a_project_bn[0][0]']  
                                                                                                  
 block3a_expand_conv (Conv2  (None, None, None, 128)      36864     ['block2b_add[0][0]']         
 D)                                                                                               
                                                                                                  
 block3a_expand_bn (BatchNo  (None, None, None, 128)      512       ['block3a_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block3a_expand_activation   (None, None, None, 128)      0         ['block3a_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block3a_project_conv (Conv  (None, None, None, 48)       6144      ['block3a_expand_activation[0]
 2D)                                                                [0]']                         
                                                                                                  
 block3a_project_bn (BatchN  (None, None, None, 48)       192       ['block3a_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block3b_expand_conv (Conv2  (None, None, None, 192)      82944     ['block3a_project_bn[0][0]']  
 D)                                                                                               
                                                                                                  
 block3b_expand_bn (BatchNo  (None, None, None, 192)      768       ['block3b_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block3b_expand_activation   (None, None, None, 192)      0         ['block3b_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block3b_project_conv (Conv  (None, None, None, 48)       9216      ['block3b_expand_activation[0]
 2D)                                                                [0]']                         
                                                                                                  
 block3b_project_bn (BatchN  (None, None, None, 48)       192       ['block3b_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block3b_drop (Dropout)      (None, None, None, 48)       0         ['block3b_project_bn[0][0]']  
                                                                                                  
 block3b_add (Add)           (None, None, None, 48)       0         ['block3b_drop[0][0]',        
                                                                     'block3a_project_bn[0][0]']  
                                                                                                  
 block4a_expand_conv (Conv2  (None, None, None, 192)      9216      ['block3b_add[0][0]']         
 D)                                                                                               
                                                                                                  
 block4a_expand_bn (BatchNo  (None, None, None, 192)      768       ['block4a_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block4a_expand_activation   (None, None, None, 192)      0         ['block4a_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block4a_dwconv2 (Depthwise  (None, None, None, 192)      1728      ['block4a_expand_activation[0]
 Conv2D)                                                            [0]']                         
                                                                                                  
 block4a_bn (BatchNormaliza  (None, None, None, 192)      768       ['block4a_dwconv2[0][0]']     
 tion)                                                                                            
                                                                                                  
 block4a_activation (Activa  (None, None, None, 192)      0         ['block4a_bn[0][0]']          
 tion)                                                                                            
                                                                                                  
 block4a_se_squeeze (Global  (None, 192)                  0         ['block4a_activation[0][0]']  
 AveragePooling2D)                                                                                
                                                                                                  
 block4a_se_reshape (Reshap  (None, 1, 1, 192)            0         ['block4a_se_squeeze[0][0]']  
 e)                                                                                               
                                                                                                  
 block4a_se_reduce (Conv2D)  (None, 1, 1, 12)             2316      ['block4a_se_reshape[0][0]']  
                                                                                                  
 block4a_se_expand (Conv2D)  (None, 1, 1, 192)            2496      ['block4a_se_reduce[0][0]']   
                                                                                                  
 block4a_se_excite (Multipl  (None, None, None, 192)      0         ['block4a_activation[0][0]',  
 y)                                                                  'block4a_se_expand[0][0]']   
                                                                                                  
 block4a_project_conv (Conv  (None, None, None, 96)       18432     ['block4a_se_excite[0][0]']   
 2D)                                                                                              
                                                                                                  
 block4a_project_bn (BatchN  (None, None, None, 96)       384       ['block4a_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block4b_expand_conv (Conv2  (None, None, None, 384)      36864     ['block4a_project_bn[0][0]']  
 D)                                                                                               
                                                                                                  
 block4b_expand_bn (BatchNo  (None, None, None, 384)      1536      ['block4b_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block4b_expand_activation   (None, None, None, 384)      0         ['block4b_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block4b_dwconv2 (Depthwise  (None, None, None, 384)      3456      ['block4b_expand_activation[0]
 Conv2D)                                                            [0]']                         
                                                                                                  
 block4b_bn (BatchNormaliza  (None, None, None, 384)      1536      ['block4b_dwconv2[0][0]']     
 tion)                                                                                            
                                                                                                  
 block4b_activation (Activa  (None, None, None, 384)      0         ['block4b_bn[0][0]']          
 tion)                                                                                            
                                                                                                  
 block4b_se_squeeze (Global  (None, 384)                  0         ['block4b_activation[0][0]']  
 AveragePooling2D)                                                                                
                                                                                                  
 block4b_se_reshape (Reshap  (None, 1, 1, 384)            0         ['block4b_se_squeeze[0][0]']  
 e)                                                                                               
                                                                                                  
 block4b_se_reduce (Conv2D)  (None, 1, 1, 24)             9240      ['block4b_se_reshape[0][0]']  
                                                                                                  
 block4b_se_expand (Conv2D)  (None, 1, 1, 384)            9600      ['block4b_se_reduce[0][0]']   
                                                                                                  
 block4b_se_excite (Multipl  (None, None, None, 384)      0         ['block4b_activation[0][0]',  
 y)                                                                  'block4b_se_expand[0][0]']   
                                                                                                  
 block4b_project_conv (Conv  (None, None, None, 96)       36864     ['block4b_se_excite[0][0]']   
 2D)                                                                                              
                                                                                                  
 block4b_project_bn (BatchN  (None, None, None, 96)       384       ['block4b_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block4b_drop (Dropout)      (None, None, None, 96)       0         ['block4b_project_bn[0][0]']  
                                                                                                  
 block4b_add (Add)           (None, None, None, 96)       0         ['block4b_drop[0][0]',        
                                                                     'block4a_project_bn[0][0]']  
                                                                                                  
 block4c_expand_conv (Conv2  (None, None, None, 384)      36864     ['block4b_add[0][0]']         
 D)                                                                                               
                                                                                                  
 block4c_expand_bn (BatchNo  (None, None, None, 384)      1536      ['block4c_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block4c_expand_activation   (None, None, None, 384)      0         ['block4c_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block4c_dwconv2 (Depthwise  (None, None, None, 384)      3456      ['block4c_expand_activation[0]
 Conv2D)                                                            [0]']                         
                                                                                                  
 block4c_bn (BatchNormaliza  (None, None, None, 384)      1536      ['block4c_dwconv2[0][0]']     
 tion)                                                                                            
                                                                                                  
 block4c_activation (Activa  (None, None, None, 384)      0         ['block4c_bn[0][0]']          
 tion)                                                                                            
                                                                                                  
 block4c_se_squeeze (Global  (None, 384)                  0         ['block4c_activation[0][0]']  
 AveragePooling2D)                                                                                
                                                                                                  
 block4c_se_reshape (Reshap  (None, 1, 1, 384)            0         ['block4c_se_squeeze[0][0]']  
 e)                                                                                               
                                                                                                  
 block4c_se_reduce (Conv2D)  (None, 1, 1, 24)             9240      ['block4c_se_reshape[0][0]']  
                                                                                                  
 block4c_se_expand (Conv2D)  (None, 1, 1, 384)            9600      ['block4c_se_reduce[0][0]']   
                                                                                                  
 block4c_se_excite (Multipl  (None, None, None, 384)      0         ['block4c_activation[0][0]',  
 y)                                                                  'block4c_se_expand[0][0]']   
                                                                                                  
 block4c_project_conv (Conv  (None, None, None, 96)       36864     ['block4c_se_excite[0][0]']   
 2D)                                                                                              
                                                                                                  
 block4c_project_bn (BatchN  (None, None, None, 96)       384       ['block4c_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block4c_drop (Dropout)      (None, None, None, 96)       0         ['block4c_project_bn[0][0]']  
                                                                                                  
 block4c_add (Add)           (None, None, None, 96)       0         ['block4c_drop[0][0]',        
                                                                     'block4b_add[0][0]']         
                                                                                                  
 block5a_expand_conv (Conv2  (None, None, None, 576)      55296     ['block4c_add[0][0]']         
 D)                                                                                               
                                                                                                  
 block5a_expand_bn (BatchNo  (None, None, None, 576)      2304      ['block5a_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block5a_expand_activation   (None, None, None, 576)      0         ['block5a_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block5a_dwconv2 (Depthwise  (None, None, None, 576)      5184      ['block5a_expand_activation[0]
 Conv2D)                                                            [0]']                         
                                                                                                  
 block5a_bn (BatchNormaliza  (None, None, None, 576)      2304      ['block5a_dwconv2[0][0]']     
 tion)                                                                                            
                                                                                                  
 block5a_activation (Activa  (None, None, None, 576)      0         ['block5a_bn[0][0]']          
 tion)                                                                                            
                                                                                                  
 block5a_se_squeeze (Global  (None, 576)                  0         ['block5a_activation[0][0]']  
 AveragePooling2D)                                                                                
                                                                                                  
 block5a_se_reshape (Reshap  (None, 1, 1, 576)            0         ['block5a_se_squeeze[0][0]']  
 e)                                                                                               
                                                                                                  
 block5a_se_reduce (Conv2D)  (None, 1, 1, 24)             13848     ['block5a_se_reshape[0][0]']  
                                                                                                  
 block5a_se_expand (Conv2D)  (None, 1, 1, 576)            14400     ['block5a_se_reduce[0][0]']   
                                                                                                  
 block5a_se_excite (Multipl  (None, None, None, 576)      0         ['block5a_activation[0][0]',  
 y)                                                                  'block5a_se_expand[0][0]']   
                                                                                                  
 block5a_project_conv (Conv  (None, None, None, 112)      64512     ['block5a_se_excite[0][0]']   
 2D)                                                                                              
                                                                                                  
 block5a_project_bn (BatchN  (None, None, None, 112)      448       ['block5a_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block5b_expand_conv (Conv2  (None, None, None, 672)      75264     ['block5a_project_bn[0][0]']  
 D)                                                                                               
                                                                                                  
 block5b_expand_bn (BatchNo  (None, None, None, 672)      2688      ['block5b_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block5b_expand_activation   (None, None, None, 672)      0         ['block5b_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block5b_dwconv2 (Depthwise  (None, None, None, 672)      6048      ['block5b_expand_activation[0]
 Conv2D)                                                            [0]']                         
                                                                                                  
 block5b_bn (BatchNormaliza  (None, None, None, 672)      2688      ['block5b_dwconv2[0][0]']     
 tion)                                                                                            
                                                                                                  
 block5b_activation (Activa  (None, None, None, 672)      0         ['block5b_bn[0][0]']          
 tion)                                                                                            
                                                                                                  
 block5b_se_squeeze (Global  (None, 672)                  0         ['block5b_activation[0][0]']  
 AveragePooling2D)                                                                                
                                                                                                  
 block5b_se_reshape (Reshap  (None, 1, 1, 672)            0         ['block5b_se_squeeze[0][0]']  
 e)                                                                                               
                                                                                                  
 block5b_se_reduce (Conv2D)  (None, 1, 1, 28)             18844     ['block5b_se_reshape[0][0]']  
                                                                                                  
 block5b_se_expand (Conv2D)  (None, 1, 1, 672)            19488     ['block5b_se_reduce[0][0]']   
                                                                                                  
 block5b_se_excite (Multipl  (None, None, None, 672)      0         ['block5b_activation[0][0]',  
 y)                                                                  'block5b_se_expand[0][0]']   
                                                                                                  
 block5b_project_conv (Conv  (None, None, None, 112)      75264     ['block5b_se_excite[0][0]']   
 2D)                                                                                              
                                                                                                  
 block5b_project_bn (BatchN  (None, None, None, 112)      448       ['block5b_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block5b_drop (Dropout)      (None, None, None, 112)      0         ['block5b_project_bn[0][0]']  
                                                                                                  
 block5b_add (Add)           (None, None, None, 112)      0         ['block5b_drop[0][0]',        
                                                                     'block5a_project_bn[0][0]']  
                                                                                                  
 block5c_expand_conv (Conv2  (None, None, None, 672)      75264     ['block5b_add[0][0]']         
 D)                                                                                               
                                                                                                  
 block5c_expand_bn (BatchNo  (None, None, None, 672)      2688      ['block5c_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block5c_expand_activation   (None, None, None, 672)      0         ['block5c_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block5c_dwconv2 (Depthwise  (None, None, None, 672)      6048      ['block5c_expand_activation[0]
 Conv2D)                                                            [0]']                         
                                                                                                  
 block5c_bn (BatchNormaliza  (None, None, None, 672)      2688      ['block5c_dwconv2[0][0]']     
 tion)                                                                                            
                                                                                                  
 block5c_activation (Activa  (None, None, None, 672)      0         ['block5c_bn[0][0]']          
 tion)                                                                                            
                                                                                                  
 block5c_se_squeeze (Global  (None, 672)                  0         ['block5c_activation[0][0]']  
 AveragePooling2D)                                                                                
                                                                                                  
 block5c_se_reshape (Reshap  (None, 1, 1, 672)            0         ['block5c_se_squeeze[0][0]']  
 e)                                                                                               
                                                                                                  
 block5c_se_reduce (Conv2D)  (None, 1, 1, 28)             18844     ['block5c_se_reshape[0][0]']  
                                                                                                  
 block5c_se_expand (Conv2D)  (None, 1, 1, 672)            19488     ['block5c_se_reduce[0][0]']   
                                                                                                  
 block5c_se_excite (Multipl  (None, None, None, 672)      0         ['block5c_activation[0][0]',  
 y)                                                                  'block5c_se_expand[0][0]']   
                                                                                                  
 block5c_project_conv (Conv  (None, None, None, 112)      75264     ['block5c_se_excite[0][0]']   
 2D)                                                                                              
                                                                                                  
 block5c_project_bn (BatchN  (None, None, None, 112)      448       ['block5c_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block5c_drop (Dropout)      (None, None, None, 112)      0         ['block5c_project_bn[0][0]']  
                                                                                                  
 block5c_add (Add)           (None, None, None, 112)      0         ['block5c_drop[0][0]',        
                                                                     'block5b_add[0][0]']         
                                                                                                  
 block5d_expand_conv (Conv2  (None, None, None, 672)      75264     ['block5c_add[0][0]']         
 D)                                                                                               
                                                                                                  
 block5d_expand_bn (BatchNo  (None, None, None, 672)      2688      ['block5d_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block5d_expand_activation   (None, None, None, 672)      0         ['block5d_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block5d_dwconv2 (Depthwise  (None, None, None, 672)      6048      ['block5d_expand_activation[0]
 Conv2D)                                                            [0]']                         
                                                                                                  
 block5d_bn (BatchNormaliza  (None, None, None, 672)      2688      ['block5d_dwconv2[0][0]']     
 tion)                                                                                            
                                                                                                  
 block5d_activation (Activa  (None, None, None, 672)      0         ['block5d_bn[0][0]']          
 tion)                                                                                            
                                                                                                  
 block5d_se_squeeze (Global  (None, 672)                  0         ['block5d_activation[0][0]']  
 AveragePooling2D)                                                                                
                                                                                                  
 block5d_se_reshape (Reshap  (None, 1, 1, 672)            0         ['block5d_se_squeeze[0][0]']  
 e)                                                                                               
                                                                                                  
 block5d_se_reduce (Conv2D)  (None, 1, 1, 28)             18844     ['block5d_se_reshape[0][0]']  
                                                                                                  
 block5d_se_expand (Conv2D)  (None, 1, 1, 672)            19488     ['block5d_se_reduce[0][0]']   
                                                                                                  
 block5d_se_excite (Multipl  (None, None, None, 672)      0         ['block5d_activation[0][0]',  
 y)                                                                  'block5d_se_expand[0][0]']   
                                                                                                  
 block5d_project_conv (Conv  (None, None, None, 112)      75264     ['block5d_se_excite[0][0]']   
 2D)                                                                                              
                                                                                                  
 block5d_project_bn (BatchN  (None, None, None, 112)      448       ['block5d_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block5d_drop (Dropout)      (None, None, None, 112)      0         ['block5d_project_bn[0][0]']  
                                                                                                  
 block5d_add (Add)           (None, None, None, 112)      0         ['block5d_drop[0][0]',        
                                                                     'block5c_add[0][0]']         
                                                                                                  
 block5e_expand_conv (Conv2  (None, None, None, 672)      75264     ['block5d_add[0][0]']         
 D)                                                                                               
                                                                                                  
 block5e_expand_bn (BatchNo  (None, None, None, 672)      2688      ['block5e_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block5e_expand_activation   (None, None, None, 672)      0         ['block5e_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block5e_dwconv2 (Depthwise  (None, None, None, 672)      6048      ['block5e_expand_activation[0]
 Conv2D)                                                            [0]']                         
                                                                                                  
 block5e_bn (BatchNormaliza  (None, None, None, 672)      2688      ['block5e_dwconv2[0][0]']     
 tion)                                                                                            
                                                                                                  
 block5e_activation (Activa  (None, None, None, 672)      0         ['block5e_bn[0][0]']          
 tion)                                                                                            
                                                                                                  
 block5e_se_squeeze (Global  (None, 672)                  0         ['block5e_activation[0][0]']  
 AveragePooling2D)                                                                                
                                                                                                  
 block5e_se_reshape (Reshap  (None, 1, 1, 672)            0         ['block5e_se_squeeze[0][0]']  
 e)                                                                                               
                                                                                                  
 block5e_se_reduce (Conv2D)  (None, 1, 1, 28)             18844     ['block5e_se_reshape[0][0]']  
                                                                                                  
 block5e_se_expand (Conv2D)  (None, 1, 1, 672)            19488     ['block5e_se_reduce[0][0]']   
                                                                                                  
 block5e_se_excite (Multipl  (None, None, None, 672)      0         ['block5e_activation[0][0]',  
 y)                                                                  'block5e_se_expand[0][0]']   
                                                                                                  
 block5e_project_conv (Conv  (None, None, None, 112)      75264     ['block5e_se_excite[0][0]']   
 2D)                                                                                              
                                                                                                  
 block5e_project_bn (BatchN  (None, None, None, 112)      448       ['block5e_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block5e_drop (Dropout)      (None, None, None, 112)      0         ['block5e_project_bn[0][0]']  
                                                                                                  
 block5e_add (Add)           (None, None, None, 112)      0         ['block5e_drop[0][0]',        
                                                                     'block5d_add[0][0]']         
                                                                                                  
 block6a_expand_conv (Conv2  (None, None, None, 672)      75264     ['block5e_add[0][0]']         
 D)                                                                                               
                                                                                                  
 block6a_expand_bn (BatchNo  (None, None, None, 672)      2688      ['block6a_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block6a_expand_activation   (None, None, None, 672)      0         ['block6a_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block6a_dwconv2 (Depthwise  (None, None, None, 672)      6048      ['block6a_expand_activation[0]
 Conv2D)                                                            [0]']                         
                                                                                                  
 block6a_bn (BatchNormaliza  (None, None, None, 672)      2688      ['block6a_dwconv2[0][0]']     
 tion)                                                                                            
                                                                                                  
 block6a_activation (Activa  (None, None, None, 672)      0         ['block6a_bn[0][0]']          
 tion)                                                                                            
                                                                                                  
 block6a_se_squeeze (Global  (None, 672)                  0         ['block6a_activation[0][0]']  
 AveragePooling2D)                                                                                
                                                                                                  
 block6a_se_reshape (Reshap  (None, 1, 1, 672)            0         ['block6a_se_squeeze[0][0]']  
 e)                                                                                               
                                                                                                  
 block6a_se_reduce (Conv2D)  (None, 1, 1, 28)             18844     ['block6a_se_reshape[0][0]']  
                                                                                                  
 block6a_se_expand (Conv2D)  (None, 1, 1, 672)            19488     ['block6a_se_reduce[0][0]']   
                                                                                                  
 block6a_se_excite (Multipl  (None, None, None, 672)      0         ['block6a_activation[0][0]',  
 y)                                                                  'block6a_se_expand[0][0]']   
                                                                                                  
 block6a_project_conv (Conv  (None, None, None, 192)      129024    ['block6a_se_excite[0][0]']   
 2D)                                                                                              
                                                                                                  
 block6a_project_bn (BatchN  (None, None, None, 192)      768       ['block6a_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block6b_expand_conv (Conv2  (None, None, None, 1152)     221184    ['block6a_project_bn[0][0]']  
 D)                                                                                               
                                                                                                  
 block6b_expand_bn (BatchNo  (None, None, None, 1152)     4608      ['block6b_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block6b_expand_activation   (None, None, None, 1152)     0         ['block6b_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block6b_dwconv2 (Depthwise  (None, None, None, 1152)     10368     ['block6b_expand_activation[0]
 Conv2D)                                                            [0]']                         
                                                                                                  
 block6b_bn (BatchNormaliza  (None, None, None, 1152)     4608      ['block6b_dwconv2[0][0]']     
 tion)                                                                                            
                                                                                                  
 block6b_activation (Activa  (None, None, None, 1152)     0         ['block6b_bn[0][0]']          
 tion)                                                                                            
                                                                                                  
 block6b_se_squeeze (Global  (None, 1152)                 0         ['block6b_activation[0][0]']  
 AveragePooling2D)                                                                                
                                                                                                  
 block6b_se_reshape (Reshap  (None, 1, 1, 1152)           0         ['block6b_se_squeeze[0][0]']  
 e)                                                                                               
                                                                                                  
 block6b_se_reduce (Conv2D)  (None, 1, 1, 48)             55344     ['block6b_se_reshape[0][0]']  
                                                                                                  
 block6b_se_expand (Conv2D)  (None, 1, 1, 1152)           56448     ['block6b_se_reduce[0][0]']   
                                                                                                  
 block6b_se_excite (Multipl  (None, None, None, 1152)     0         ['block6b_activation[0][0]',  
 y)                                                                  'block6b_se_expand[0][0]']   
                                                                                                  
 block6b_project_conv (Conv  (None, None, None, 192)      221184    ['block6b_se_excite[0][0]']   
 2D)                                                                                              
                                                                                                  
 block6b_project_bn (BatchN  (None, None, None, 192)      768       ['block6b_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block6b_drop (Dropout)      (None, None, None, 192)      0         ['block6b_project_bn[0][0]']  
                                                                                                  
 block6b_add (Add)           (None, None, None, 192)      0         ['block6b_drop[0][0]',        
                                                                     'block6a_project_bn[0][0]']  
                                                                                                  
 block6c_expand_conv (Conv2  (None, None, None, 1152)     221184    ['block6b_add[0][0]']         
 D)                                                                                               
                                                                                                  
 block6c_expand_bn (BatchNo  (None, None, None, 1152)     4608      ['block6c_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block6c_expand_activation   (None, None, None, 1152)     0         ['block6c_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block6c_dwconv2 (Depthwise  (None, None, None, 1152)     10368     ['block6c_expand_activation[0]
 Conv2D)                                                            [0]']                         
                                                                                                  
 block6c_bn (BatchNormaliza  (None, None, None, 1152)     4608      ['block6c_dwconv2[0][0]']     
 tion)                                                                                            
                                                                                                  
 block6c_activation (Activa  (None, None, None, 1152)     0         ['block6c_bn[0][0]']          
 tion)                                                                                            
                                                                                                  
 block6c_se_squeeze (Global  (None, 1152)                 0         ['block6c_activation[0][0]']  
 AveragePooling2D)                                                                                
                                                                                                  
 block6c_se_reshape (Reshap  (None, 1, 1, 1152)           0         ['block6c_se_squeeze[0][0]']  
 e)                                                                                               
                                                                                                  
 block6c_se_reduce (Conv2D)  (None, 1, 1, 48)             55344     ['block6c_se_reshape[0][0]']  
                                                                                                  
 block6c_se_expand (Conv2D)  (None, 1, 1, 1152)           56448     ['block6c_se_reduce[0][0]']   
                                                                                                  
 block6c_se_excite (Multipl  (None, None, None, 1152)     0         ['block6c_activation[0][0]',  
 y)                                                                  'block6c_se_expand[0][0]']   
                                                                                                  
 block6c_project_conv (Conv  (None, None, None, 192)      221184    ['block6c_se_excite[0][0]']   
 2D)                                                                                              
                                                                                                  
 block6c_project_bn (BatchN  (None, None, None, 192)      768       ['block6c_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block6c_drop (Dropout)      (None, None, None, 192)      0         ['block6c_project_bn[0][0]']  
                                                                                                  
 block6c_add (Add)           (None, None, None, 192)      0         ['block6c_drop[0][0]',        
                                                                     'block6b_add[0][0]']         
                                                                                                  
 block6d_expand_conv (Conv2  (None, None, None, 1152)     221184    ['block6c_add[0][0]']         
 D)                                                                                               
                                                                                                  
 block6d_expand_bn (BatchNo  (None, None, None, 1152)     4608      ['block6d_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block6d_expand_activation   (None, None, None, 1152)     0         ['block6d_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block6d_dwconv2 (Depthwise  (None, None, None, 1152)     10368     ['block6d_expand_activation[0]
 Conv2D)                                                            [0]']                         
                                                                                                  
 block6d_bn (BatchNormaliza  (None, None, None, 1152)     4608      ['block6d_dwconv2[0][0]']     
 tion)                                                                                            
                                                                                                  
 block6d_activation (Activa  (None, None, None, 1152)     0         ['block6d_bn[0][0]']          
 tion)                                                                                            
                                                                                                  
 block6d_se_squeeze (Global  (None, 1152)                 0         ['block6d_activation[0][0]']  
 AveragePooling2D)                                                                                
                                                                                                  
 block6d_se_reshape (Reshap  (None, 1, 1, 1152)           0         ['block6d_se_squeeze[0][0]']  
 e)                                                                                               
                                                                                                  
 block6d_se_reduce (Conv2D)  (None, 1, 1, 48)             55344     ['block6d_se_reshape[0][0]']  
                                                                                                  
 block6d_se_expand (Conv2D)  (None, 1, 1, 1152)           56448     ['block6d_se_reduce[0][0]']   
                                                                                                  
 block6d_se_excite (Multipl  (None, None, None, 1152)     0         ['block6d_activation[0][0]',  
 y)                                                                  'block6d_se_expand[0][0]']   
                                                                                                  
 block6d_project_conv (Conv  (None, None, None, 192)      221184    ['block6d_se_excite[0][0]']   
 2D)                                                                                              
                                                                                                  
 block6d_project_bn (BatchN  (None, None, None, 192)      768       ['block6d_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block6d_drop (Dropout)      (None, None, None, 192)      0         ['block6d_project_bn[0][0]']  
                                                                                                  
 block6d_add (Add)           (None, None, None, 192)      0         ['block6d_drop[0][0]',        
                                                                     'block6c_add[0][0]']         
                                                                                                  
 block6e_expand_conv (Conv2  (None, None, None, 1152)     221184    ['block6d_add[0][0]']         
 D)                                                                                               
                                                                                                  
 block6e_expand_bn (BatchNo  (None, None, None, 1152)     4608      ['block6e_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block6e_expand_activation   (None, None, None, 1152)     0         ['block6e_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block6e_dwconv2 (Depthwise  (None, None, None, 1152)     10368     ['block6e_expand_activation[0]
 Conv2D)                                                            [0]']                         
                                                                                                  
 block6e_bn (BatchNormaliza  (None, None, None, 1152)     4608      ['block6e_dwconv2[0][0]']     
 tion)                                                                                            
                                                                                                  
 block6e_activation (Activa  (None, None, None, 1152)     0         ['block6e_bn[0][0]']          
 tion)                                                                                            
                                                                                                  
 block6e_se_squeeze (Global  (None, 1152)                 0         ['block6e_activation[0][0]']  
 AveragePooling2D)                                                                                
                                                                                                  
 block6e_se_reshape (Reshap  (None, 1, 1, 1152)           0         ['block6e_se_squeeze[0][0]']  
 e)                                                                                               
                                                                                                  
 block6e_se_reduce (Conv2D)  (None, 1, 1, 48)             55344     ['block6e_se_reshape[0][0]']  
                                                                                                  
 block6e_se_expand (Conv2D)  (None, 1, 1, 1152)           56448     ['block6e_se_reduce[0][0]']   
                                                                                                  
 block6e_se_excite (Multipl  (None, None, None, 1152)     0         ['block6e_activation[0][0]',  
 y)                                                                  'block6e_se_expand[0][0]']   
                                                                                                  
 block6e_project_conv (Conv  (None, None, None, 192)      221184    ['block6e_se_excite[0][0]']   
 2D)                                                                                              
                                                                                                  
 block6e_project_bn (BatchN  (None, None, None, 192)      768       ['block6e_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block6e_drop (Dropout)      (None, None, None, 192)      0         ['block6e_project_bn[0][0]']  
                                                                                                  
 block6e_add (Add)           (None, None, None, 192)      0         ['block6e_drop[0][0]',        
                                                                     'block6d_add[0][0]']         
                                                                                                  
 block6f_expand_conv (Conv2  (None, None, None, 1152)     221184    ['block6e_add[0][0]']         
 D)                                                                                               
                                                                                                  
 block6f_expand_bn (BatchNo  (None, None, None, 1152)     4608      ['block6f_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block6f_expand_activation   (None, None, None, 1152)     0         ['block6f_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block6f_dwconv2 (Depthwise  (None, None, None, 1152)     10368     ['block6f_expand_activation[0]
 Conv2D)                                                            [0]']                         
                                                                                                  
 block6f_bn (BatchNormaliza  (None, None, None, 1152)     4608      ['block6f_dwconv2[0][0]']     
 tion)                                                                                            
                                                                                                  
 block6f_activation (Activa  (None, None, None, 1152)     0         ['block6f_bn[0][0]']          
 tion)                                                                                            
                                                                                                  
 block6f_se_squeeze (Global  (None, 1152)                 0         ['block6f_activation[0][0]']  
 AveragePooling2D)                                                                                
                                                                                                  
 block6f_se_reshape (Reshap  (None, 1, 1, 1152)           0         ['block6f_se_squeeze[0][0]']  
 e)                                                                                               
                                                                                                  
 block6f_se_reduce (Conv2D)  (None, 1, 1, 48)             55344     ['block6f_se_reshape[0][0]']  
                                                                                                  
 block6f_se_expand (Conv2D)  (None, 1, 1, 1152)           56448     ['block6f_se_reduce[0][0]']   
                                                                                                  
 block6f_se_excite (Multipl  (None, None, None, 1152)     0         ['block6f_activation[0][0]',  
 y)                                                                  'block6f_se_expand[0][0]']   
                                                                                                  
 block6f_project_conv (Conv  (None, None, None, 192)      221184    ['block6f_se_excite[0][0]']   
 2D)                                                                                              
                                                                                                  
 block6f_project_bn (BatchN  (None, None, None, 192)      768       ['block6f_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block6f_drop (Dropout)      (None, None, None, 192)      0         ['block6f_project_bn[0][0]']  
                                                                                                  
 block6f_add (Add)           (None, None, None, 192)      0         ['block6f_drop[0][0]',        
                                                                     'block6e_add[0][0]']         
                                                                                                  
 block6g_expand_conv (Conv2  (None, None, None, 1152)     221184    ['block6f_add[0][0]']         
 D)                                                                                               
                                                                                                  
 block6g_expand_bn (BatchNo  (None, None, None, 1152)     4608      ['block6g_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block6g_expand_activation   (None, None, None, 1152)     0         ['block6g_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block6g_dwconv2 (Depthwise  (None, None, None, 1152)     10368     ['block6g_expand_activation[0]
 Conv2D)                                                            [0]']                         
                                                                                                  
 block6g_bn (BatchNormaliza  (None, None, None, 1152)     4608      ['block6g_dwconv2[0][0]']     
 tion)                                                                                            
                                                                                                  
 block6g_activation (Activa  (None, None, None, 1152)     0         ['block6g_bn[0][0]']          
 tion)                                                                                            
                                                                                                  
 block6g_se_squeeze (Global  (None, 1152)                 0         ['block6g_activation[0][0]']  
 AveragePooling2D)                                                                                
                                                                                                  
 block6g_se_reshape (Reshap  (None, 1, 1, 1152)           0         ['block6g_se_squeeze[0][0]']  
 e)                                                                                               
                                                                                                  
 block6g_se_reduce (Conv2D)  (None, 1, 1, 48)             55344     ['block6g_se_reshape[0][0]']  
                                                                                                  
 block6g_se_expand (Conv2D)  (None, 1, 1, 1152)           56448     ['block6g_se_reduce[0][0]']   
                                                                                                  
 block6g_se_excite (Multipl  (None, None, None, 1152)     0         ['block6g_activation[0][0]',  
 y)                                                                  'block6g_se_expand[0][0]']   
                                                                                                  
 block6g_project_conv (Conv  (None, None, None, 192)      221184    ['block6g_se_excite[0][0]']   
 2D)                                                                                              
                                                                                                  
 block6g_project_bn (BatchN  (None, None, None, 192)      768       ['block6g_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block6g_drop (Dropout)      (None, None, None, 192)      0         ['block6g_project_bn[0][0]']  
                                                                                                  
 block6g_add (Add)           (None, None, None, 192)      0         ['block6g_drop[0][0]',        
                                                                     'block6f_add[0][0]']         
                                                                                                  
 block6h_expand_conv (Conv2  (None, None, None, 1152)     221184    ['block6g_add[0][0]']         
 D)                                                                                               
                                                                                                  
 block6h_expand_bn (BatchNo  (None, None, None, 1152)     4608      ['block6h_expand_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 block6h_expand_activation   (None, None, None, 1152)     0         ['block6h_expand_bn[0][0]']   
 (Activation)                                                                                     
                                                                                                  
 block6h_dwconv2 (Depthwise  (None, None, None, 1152)     10368     ['block6h_expand_activation[0]
 Conv2D)                                                            [0]']                         
                                                                                                  
 block6h_bn (BatchNormaliza  (None, None, None, 1152)     4608      ['block6h_dwconv2[0][0]']     
 tion)                                                                                            
                                                                                                  
 block6h_activation (Activa  (None, None, None, 1152)     0         ['block6h_bn[0][0]']          
 tion)                                                                                            
                                                                                                  
 block6h_se_squeeze (Global  (None, 1152)                 0         ['block6h_activation[0][0]']  
 AveragePooling2D)                                                                                
                                                                                                  
 block6h_se_reshape (Reshap  (None, 1, 1, 1152)           0         ['block6h_se_squeeze[0][0]']  
 e)                                                                                               
                                                                                                  
 block6h_se_reduce (Conv2D)  (None, 1, 1, 48)             55344     ['block6h_se_reshape[0][0]']  
                                                                                                  
 block6h_se_expand (Conv2D)  (None, 1, 1, 1152)           56448     ['block6h_se_reduce[0][0]']   
                                                                                                  
 block6h_se_excite (Multipl  (None, None, None, 1152)     0         ['block6h_activation[0][0]',  
 y)                                                                  'block6h_se_expand[0][0]']   
                                                                                                  
 block6h_project_conv (Conv  (None, None, None, 192)      221184    ['block6h_se_excite[0][0]']   
 2D)                                                                                              
                                                                                                  
 block6h_project_bn (BatchN  (None, None, None, 192)      768       ['block6h_project_conv[0][0]']
 ormalization)                                                                                    
                                                                                                  
 block6h_drop (Dropout)      (None, None, None, 192)      0         ['block6h_project_bn[0][0]']  
                                                                                                  
 block6h_add (Add)           (None, None, None, 192)      0         ['block6h_drop[0][0]',        
                                                                     'block6g_add[0][0]']         
                                                                                                  
 top_conv (Conv2D)           (None, None, None, 1280)     245760    ['block6h_add[0][0]']         
                                                                                                  
 top_bn (BatchNormalization  (None, None, None, 1280)     5120      ['top_conv[0][0]']            
 )                                                                                                
                                                                                                  
 top_activation (Activation  (None, None, None, 1280)     0         ['top_bn[0][0]']              
 )                                                                                                
                                                                                                  
==================================================================================================
Total params: 5919312 (22.58 MB)
Trainable params: 0 (0.00 Byte)
Non-trainable params: 5919312 (22.58 MB)
__________________________________________________________________________________________________
```
- **Final Training Loss:** 1.1487236022949219
- **Final Validation Loss:** 1.1117841005325317
- **Final Training Accuracy:** 0.6041666865348816
- **Final Validation Accuracy:** 0.612500011920929
- **Highest Training Accuracy:** 0.6177083253860474
- **Highest Validation Accuracy:** 0.6458333134651184
  
![resnet](https://github.com/SanthoshToorpu/Audio-Classififcation/assets/90833739/0db414bd-8ac1-4f93-9fbc-13b1210397db)
![eh](https://github.com/SanthoshToorpu/Audio-Classififcation/assets/90833739/d0359aab-1b54-4f72-ad85-989ed81e02e3)

