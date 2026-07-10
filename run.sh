
###
 # @Description: 
 # @Date: 2024-11-23 13:12:21
 # @LastEditTime: 2024-11-23 15:38:16
 # @FilePath: /QJ/E3Diff/run.sh
### 

# stage 1 training for SAR2EO dataset
python main.py --config config/SAR2EO_256_s1.json

# stage 2 training for SAR2EO dataset
python main.py --config config/SAR2EO_256_s1_1step.json

# stage 2 validation for SAR2EO dataset
python main.py --config config/SAR2EO_256_s2_test.json --phase val  --seed 1


# stage 1 training for sen12 dataset
python main.py --config config/SEN12_256_s1.json

# stage 2 training for sen12 dataset
python main.py --config config/SEN12_256_s2_1step.json


# stage 2 validation for sen12 dataset
python main.py --config 'config/SEN12_256_s2_test.json' --phase val  --seed 1
