# nohup python -u main_train.py --cuda 1 --datadir ../data/ --MIMIC 3 --model_name Model_mimic3_test_for_camred --encoder main --seed 312 >see_if_right3.out &
# nohup python -u main_train.py --cuda 1 --datadir ../data/ --MIMIC 4 --model_name Model_mimic4_test_for_camred --encoder main --seed 312 >see_if_right4.out &


# nohup python -u main_train.py --cuda 2 --datadir ../data/ --MIMIC 3 --model_name Model_mimic3_check_DDI_enc --encoder main --seed 312 --ddi_encoding --gnn_type gat --num_layer 10 --p_or_m minus >DDI_see_if_right3.out &
# nohup python -u main_train.py --cuda 2 --datadir ../data/ --MIMIC 4 --model_name Model_mimic4_check_DDI_enc --encoder main --seed 312 --ddi_encoding --gnn_type gat --num_layer 10 --p_or_m minus >DDI_see_if_right4.out &

# nohup python -u main_train.py --cuda 1 --datadir ../data/ --MIMIC 3 --model_name Model_mimic3_test_for_camred --encoder main --seed 312 --Test --resume_path saved/Model_mimic3_test_for_camred/best.model >test_see_if_right3.out &
# nohup python -u main_train.py --cuda 1 --datadir ../data/ --MIMIC 4 --model_name Model_mimic4_test_for_camred --encoder main --seed 312 --Test --resume_path saved/Model_mimic4_test_for_camred/best.model >test_see_if_right4.out &

nohup python -u main_train.py --cuda 2 --datadir ../data/ --MIMIC 3 --model_name Model_mimic3_check_DDI_enc --encoder main --seed 312 --ddi_encoding --gnn_type gat --num_layer 10 --p_or_m minus --Test --resume_path saved/Model_mimic3_check_DDI_enc/best.model >test_DDI_see_if_right3.out &
nohup python -u main_train.py --cuda 2 --datadir ../data/ --MIMIC 4 --model_name Model_mimic4_check_DDI_enc --encoder main --seed 312 --ddi_encoding --gnn_type gat --num_layer 10 --p_or_m minus --Test --resume_path saved/Model_mimic4_check_DDI_enc/best.model >test_DDI_see_if_right4.out &
