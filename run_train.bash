# accelerate launch train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/mol2cap/" --task "mol2cap" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --disable_lora
# accelerate launch train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/cap2mol/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --disable_lora
# accelerate launch train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/mol2capR/" --task "mol2cap" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot 2
# accelerate launch train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/cap2molR/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot 2 --learning_rate 2e-8
# accelerate launch train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/instruct/" --task "instruct" --micro_batch_size 4 --batch_size 32 --num_epochs 20 --retrieval --disable_lora --n_shot 2 --train_on_inputs
# python test.py --adapter_path "./ckp/galactica-125M/mol2cap/checkpoint-8000/" --task "mol2cap" --output_dir "./predictions/galactica-125M/" --disable_lora --batch_infer
# python test.py --adapter_path "./ckp/galactica-125M/cap2mol/checkpoint-8000/" --task "cap2mol" --output_dir "./predictions/galactica-125M/" --disable_lora --batch_infer
# python test.py --adapter_path "./ckp/galactica-125M/mol2capR/checkpoint-8000/" --task "mol2cap" --output_dir "./predictions/galactica-125M-Retrieval/" --retrieval --disable_lora --batch_infer
# python test.py --adapter_path "./ckp/galactica-125M/cap2molR/checkpoint-4000/" --task "cap2mol" --output_dir "./test_predictions/galactica-125M-R/" --retrieval --disable_lora --batch_infer
# python test.py --adapter_path "./ckp/galactica-125M/cap2molR/checkpoint-8000/" --task "cap2mol" --output_dir "./test_predictions/galactica-125M-R/" --retrieval --disable_lora --batch_infer
# legacy bash script

# python test.py --base_model "facebook/galactica-125m" --adapter_path "./ckp/galactica-125M/instruct/checkpoint-32000/" --task "cap2mol" --output_dir "./predictions/galactica-125M-instruct/" --retrieval --disable_lora --batch_infer --n_shot 2
# python test.py --base_model "facebook/galactica-125m" --adapter_path "./ckp/galactica-125M/instruct/checkpoint-32000/" --task "mol2cap" --output_dir "./predictions/galactica-125M-instruct/" --retrieval --disable_lora --batch_infer --n_shot 2

# grid search for galactica-125M
# for lr in 1e-4 2e-4 5e-4 1e-3
# do
#     for n_shot in 1 2
#     do
#         accelerate launch train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/cap2mol-$lr-$n_shot/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr
#         python test.py --adapter_path "./ckp/galactica-125M/cap2mol-$lr-$n_shot/checkpoint-8000/" --task "cap2mol" --output_dir "./predictions/galactica-125M-R-$lr-$n_shot/" --retrieval --disable_lora --batch_infer --n_shot $n_shot
#     done
# done
# for lr in 2e-4
# do
#     for n_shot in 2
#     do
#         accelerate launch train_ppo.py --base_model "../vicuna/llama2-7b-chat-hf" --output_dir "./ckp/llama2-7b/cap2mol-$lr-$n_shot/" --task "cap2mol" --micro_batch_size 1 --batch_size 32 --num_epochs 3 --train_on_inputs --learning_rate $lr --n_shot $n_shot --retrieval --int8 --fp16 --save_interval 800
#         python test.py --base_model "../vicuna/llama2-7b-chat-hf" --adapter_path "./ckp/llama2-7b/cap2mol-$lr-$n_shot/checkpoint-2400/" --task "cap2mol" --output_dir "./predictions/llama2-7b-$lr-$n_shot/" --n_shot $n_shot --retrieval --fp16
#         accelerate launch train_ppo.py --base_model "../vicuna/llama2-7b-chat-hf" --output_dir "./ckp/llama2-7b/mol2cap-$lr-$n_shot/" --task "mol2cap" --micro_batch_size 1 --batch_size 32 --num_epochs 3 --train_on_inputs --learning_rate $lr --n_shot $n_shot --retrieval --int8 --fp16 --save_interval 800
#         python test.py --base_model "../vicuna/llama2-7b-chat-hf" --adapter_path "./ckp/llama2-7b/mol2cap-$lr-$n_shot/checkpoint-2400/" --task "mol2cap" --output_dir "./predictions/llama2-7b-$lr-$n_shot/" --n_shot $n_shot --retrieval --fp16
        
#         accelerate launch train_ppo.py --base_model "../vicuna/llama2-7b-chat-hf" --output_dir "./ckp/llama2-7b/mol2cap/" --task "mol2cap" --micro_batch_size 1 --batch_size 32 --num_epochs 3 --train_on_inputs --learning_rate $lr --int8 --fp16 --save_interval 800
#         python test.py --base_model "../vicuna/llama2-7b-chat-hf" --adapter_path "./ckp/llama2-7b/mol2cap/checkpoint-2400/" --task "mol2cap" --output_dir "./predictions/llama2-7b/" --fp16
#         accelerate launch train_ppo.py --base_model "../vicuna/llama2-7b-chat-hf" --output_dir "./ckp/llama2-7b/cap2mol/" --task "cap2mol" --micro_batch_size 1 --batch_size 32 --num_epochs 3 --train_on_inputs --learning_rate $lr --int8 --fp16 --save_interval 800
#         python test.py --base_model "../vicuna/llama2-7b-chat-hf" --adapter_path "./ckp/llama2-7b/cap2mol/checkpoint-2400/" --task "cap2mol" --output_dir "./predictions/llama2-7b/" --fp16
#     done
# done


# for lr in 2e-4
# do
#     for n_shot in 1
#     do
#         # accelerate launch train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/cap2mol-$lr-$n_shot/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr
#         python test.py --adapter_path "./ckp/galactica-125M/cap2mol-$lr-$n_shot/checkpoint-16000/" --task "cap2mol" --output_dir "./predictions/galactica-125M-R-$lr-$n_shot/" --retrieval --disable_lora --batch_infer --n_shot $n_shot
#     done
# done

# for lr in 2e-4
# do
#     for n_shot in 2
#     do
#         for len in 1024
#         do
#             CUDA_VISIBLE_DEVICES=1 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/mol2cap-$lr-$n_shot-$len/" --task "mol2cap" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --bucket_sampling
#             CUDA_VISIBLE_DEVICES=1 python test.py --adapter_path "./ckp/galactica-125M/mol2cap-$lr-$n_shot-$len/checkpoint-8000/" --task "mol2cap" --output_dir "./new_predictions/galactica-125M-B-$lr-$n_shot-$len/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --bucket_sampling
#         done
#     done
# done

# for seed in 77
# do
#     for lr in 2e-4
#     do
#         for n_shot in 2
#         do
#             for len in 1024
#             do
#                 CUDA_VISIBLE_DEVICES=3 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/mol2cap-$lr-$n_shot-$len-$seed/" --task "mol2cap" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --bucket_sampling --seed $seed
#                 CUDA_VISIBLE_DEVICES=3 python test.py --adapter_path "./ckp/galactica-125M/mol2cap-$lr-$n_shot-$len-$seed/checkpoint-8000/" --task "mol2cap" --output_dir "./new_predictions/galactica-125M-B-$lr-$n_shot-$len-$seed/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --bucket_sampling --seed $seed
            
#                 CUDA_VISIBLE_DEVICES=3 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len-$seed/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --bucket_sampling --seed $seed
#                 CUDA_VISIBLE_DEVICES=3 python test.py --adapter_path "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len-$seed/checkpoint-8000/" --task "cap2mol" --output_dir "./rebuttal_predictions/galactica-125M-B-$lr-$n_shot-$len-$seed/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --bucket_sampling --seed $seed
#             done
#         done
#     done
# done

# for seed in 128
# do
#     for lr in 2e-4
#     do
#         for n_shot in 2
#         do
#             for len in 1024
#             do
#                 CUDA_VISIBLE_DEVICES=3 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len-$seed/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --bucket_sampling --seed $seed
#                 CUDA_VISIBLE_DEVICES=3 python test.py --adapter_path "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len-$seed/checkpoint-8000/" --task "cap2mol" --output_dir "./rebuttal_predictions/galactica-125M-B-$lr-$n_shot-$len-$seed/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --bucket_sampling --seed $seed
#             done
#         done
#     done
# done

# for lr in 2e-4
# do
#     for n_shot in 2
#     do
#         for len in 1024
#         do
#             CUDA_VISIBLE_DEVICES=1 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/mol2cap-$lr-$n_shot-$len-MR/" --task "mol2cap" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk
#             CUDA_VISIBLE_DEVICES=1 python test.py --adapter_path "./ckp/galactica-125M/mol2cap-$lr-$n_shot-$len-MR/checkpoint-8000/" --task "mol2cap" --output_dir "./rebuttal_predictions/galactica-125M-MR-$lr-$n_shot-$len/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --random_walk
#         done
#     done
# done


# for lr in 2e-4
# do
#     for n_shot in 2
#     do
#         for len in 1024
#         do
#             CUDA_VISIBLE_DEVICES=1 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len-MR/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk 
#             CUDA_VISIBLE_DEVICES=1 python test.py --adapter_path "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len-MR/checkpoint-8000/" --task "cap2mol" --output_dir "./rebuttal_predictions/galactica-125M-MR-$lr-$n_shot-$len/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --random_walk 

#             CUDA_VISIBLE_DEVICES=1 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len-RR/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk --m2c_method 'random'
#             CUDA_VISIBLE_DEVICES=1 python test.py --adapter_path "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len-RR/checkpoint-8000/" --task "cap2mol" --output_dir "./rebuttal_predictions/galactica-125M-RR-$lr-$n_shot-$len/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --random_walk --m2c_method 'random'

#             CUDA_VISIBLE_DEVICES=1 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len-FR/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk --m2c_method 'morgan'
#             CUDA_VISIBLE_DEVICES=1 python test.py --adapter_path "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len-FR/checkpoint-8000/" --task "cap2mol" --output_dir "./rebuttal_predictions/galactica-125M-FR-$lr-$n_shot-$len/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --random_walk --m2c_method 'morgan'
#         done
#     done
# done

            
# for lr in 2e-4
# do
#     for n_shot in 4
#     do
#         for len in 1536
#         do
#             # CUDA_VISIBLE_DEVICES=3 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk
#             CUDA_VISIBLE_DEVICES=3 python test.py --adapter_path "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len/checkpoint-8000/" --task "cap2mol" --output_dir "./new_predictions/galactica-125M-R-$lr-$n_shot-$len/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --random_walk

#             # CUDA_VISIBLE_DEVICES=3 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/mol2cap-$lr-$n_shot-$len/" --task "mol2cap" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk
#             CUDA_VISIBLE_DEVICES=3 python test.py --adapter_path "./ckp/galactica-125M/mol2cap-$lr-$n_shot-$len/checkpoint-8000/" --task "mol2cap" --output_dir "./new_predictions/galactica-125M-R-$lr-$n_shot-$len/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --random_walk
#         done
#     done
# done

# for lr in 2e-4
# do
#     for n_shot in 1
#     do
#         for len in 2048
#         do
#             # CUDA_VISIBLE_DEVICES=2 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk
#             # CUDA_VISIBLE_DEVICES=2 python test.py --adapter_path "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len/checkpoint-8000/" --task "cap2mol" --output_dir "./new_predictions/galactica-125M-R-$lr-$n_shot-$len/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --random_walk

#             CUDA_VISIBLE_DEVICES=2 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/mol2cap-$lr-$n_shot-$len/" --task "mol2cap" --micro_batch_size 1 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk
#             CUDA_VISIBLE_DEVICES=2 python test.py --adapter_path "./ckp/galactica-125M/mol2cap-$lr-$n_shot-$len/checkpoint-8000/" --task "mol2cap" --output_dir "./new_predictions/galactica-125M-R-$lr-$n_shot-$len/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --random_walk
#         done
#     done
# done

# for lr in 2e-4
# do
#     for n_shot in 2
#     do
#         for len in 1024
#         do
#             # CUDA_VISIBLE_DEVICES=2 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/mol2cap-$lr-$n_shot-$len/" --task "mol2cap" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk
#             CUDA_VISIBLE_DEVICES=2 python test.py --adapter_path "./ckp/galactica-125M/mol2cap-$lr-$n_shot-$len/checkpoint-8000/" --task "mol2cap" --output_dir "./new_predictions/galactica-125M-R-$lr-$n_shot-$len/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --random_walk
       
#             # CUDA_VISIBLE_DEVICES=2 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len-Random/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk --c2m_method "random"
#             CUDA_VISIBLE_DEVICES=2 python test.py --adapter_path "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len-Random/checkpoint-8000/" --task "cap2mol" --output_dir "./new_predictions/galactica-125M-R-$lr-$n_shot-$len-Random/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --random_walk --c2m_method "random"

#             # CUDA_VISIBLE_DEVICES=2 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/mol2cap-$lr-$n_shot-$len-Random/" --task "mol2cap" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk --m2c_method "random"
#             CUDA_VISIBLE_DEVICES=2 python test.py --adapter_path "./ckp/galactica-125M/mol2cap-$lr-$n_shot-$len-Random/checkpoint-8000/" --task "mol2cap" --output_dir "./new_predictions/galactica-125M-R-$lr-$n_shot-$len-Random/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --random_walk --m2c_method "random"

#             # CUDA_VISIBLE_DEVICES=2 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len-sbert/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk --c2m_method "sbert"
#             CUDA_VISIBLE_DEVICES=2 python test.py --adapter_path "./ckp/galactica-125M/cap2mol-$lr-$n_shot-$len-sbert/checkpoint-8000/" --task "cap2mol" --output_dir "./new_predictions/galactica-125M-R-$lr-$n_shot-$len-sbert/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --random_walk --c2m_method "sbert"

#             # CUDA_VISIBLE_DEVICES=2 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/mol2cap-$lr-$n_shot-$len-morgan/" --task "mol2cap" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk --m2c_method "morgan"
#             CUDA_VISIBLE_DEVICES=2 python test.py --adapter_path "./ckp/galactica-125M/mol2cap-$lr-$n_shot-$len-morgan/checkpoint-8000/" --task "mol2cap" --output_dir "./new_predictions/galactica-125M-R-$lr-$n_shot-$len-morgan/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --random_walk --m2c_method "morgan"
#         done
#     done
# done


# for lr in 2e-4
# do
#     for n_shot in 2 4
#     do
#         for len in 2048
#         do
#             python train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/bace-R-$lr-$n_shot-$len/" --task "bace" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk
#             python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/galactica-125M/bace-R-$lr-$n_shot-$len/checkpoint-370/" --task "bace" --output_dir "./moleculenet_predictions/galactica-125M-R-$lr-$n_shot-$len/" --retrieval --disable_lora --n_shot $n_shot --reverse --random_walk
#             python naive_test.py --raw_folder "./data/MoleculeNet/" --target_folder "moleculenet_predictions/" --model "galactica-125M-R-$lr-$n_shot-$len" --ckp 370 --task bace
#         done
#     done
# done

# python train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/bace/" --task "bace" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --disable_lora --train_on_inputs --learning_rate 2e-4 --cutoff_len 2048
# python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/galactica-125M/bace/checkpoint-370/" --task "bace" --output_dir "./moleculenet_predictions/galactica-125M/" --disable_lora
# python naive_test.py --raw_folder "./data/MoleculeNet/" --target_folder "moleculenet_predictions/" --model "galactica-125M" --ckp 370 --task bace

# for lr in 2e-4
# do
#     for n_shot in 2 4
#     do
#         for len in 2048
#         do
#             python train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/bbbp-R-$lr-$n_shot-$len/" --task "bbbp" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk
#             python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/galactica-125M/bbbp-R-$lr-$n_shot-$len/checkpoint-510/" --task "bbbp" --output_dir "./moleculenet_predictions/galactica-125M-R-$lr-$n_shot-$len/" --retrieval --disable_lora --n_shot $n_shot --reverse --random_walk
#             python naive_test.py --raw_folder "./data/MoleculeNet/" --target_folder "moleculenet_predictions/" --model "galactica-125M-R-$lr-$n_shot-$len" --ckp 510 --task bbbp
#         done
#     done
# done

# python train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/bbbp/" --task "bbbp" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --disable_lora --train_on_inputs --learning_rate 2e-4 --cutoff_len 2048
# python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/galactica-125M/bbbp/checkpoint-510/" --task "bbbp" --output_dir "./moleculenet_predictions/galactica-125M/" --disable_lora
# python naive_test.py --raw_folder "./data/MoleculeNet/" --target_folder "moleculenet_predictions/" --model "galactica-125M" --ckp 510 --task bbbp

# for lr in 2e-4
# do
#     for n_shot in 2 4
#     do
#         for len in 2048
#         do
#             # CUDA_VISIBLE_DEVICES=5 python train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/tox21-R-$lr-$n_shot-$len/" --task "tox21" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk
#             CUDA_VISIBLE_DEVICES=5 python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/galactica-125M/tox21-R-$lr-$n_shot-$len/checkpoint-19420/" --task "tox21" --output_dir "./moleculenet_predictions/galactica-125M-R-$lr-$n_shot-$len/" --retrieval --disable_lora --n_shot $n_shot --reverse --random_walk --batch_infer
#         done
#     done
# done

# CUDA_VISIBLE_DEVICES=5 python train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/tox21/" --task "tox21" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --disable_lora --train_on_inputs --learning_rate 2e-4 --cutoff_len 2048
# CUDA_VISIBLE_DEVICES=5 python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/galactica-125M/tox21/checkpoint-19420/" --task "tox21" --output_dir "./moleculenet_predictions/galactica-125M/" --disable_lora

# for lr in 2e-4
# do
#     for n_shot in 2 4
#     do
#         for len in 2048
#         do
#             CUDA_VISIBLE_DEVICES=5 python train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/sider-R-$lr-$n_shot-$len/" --task "sider" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk
#             CUDA_VISIBLE_DEVICES=5 python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/galactica-125M/sider-R-$lr-$n_shot-$len/checkpoint-9620/" --task "sider" --output_dir "./moleculenet_predictions/galactica-125M-R-$lr-$n_shot-$len/" --retrieval --disable_lora --n_shot $n_shot --reverse --random_walk --batch_infer
#         done
#     done
# done

# CUDA_VISIBLE_DEVICES=5 python train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/sider/" --task "sider" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --disable_lora --train_on_inputs --learning_rate 2e-4 --cutoff_len 2048
# CUDA_VISIBLE_DEVICES=5 python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/galactica-125M/sider/checkpoint-9620/" --task "sider" --output_dir "./moleculenet_predictions/galactica-125M/" --disable_lora

# for lr in 2e-4
# do
#     for n_shot in 2 4
#     do
#         for len in 2048
#         do
#             CUDA_VISIBLE_DEVICES=5 python train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/hiv-R-$lr-$n_shot-$len/" --task "hiv" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk
#             CUDA_VISIBLE_DEVICES=5 python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/galactica-125M/hiv-R-$lr-$n_shot-$len/checkpoint-10280/" --task "hiv" --output_dir "./moleculenet_predictions/galactica-125M-R-$lr-$n_shot-$len/" --retrieval --disable_lora --n_shot $n_shot --reverse --random_walk --batch_infer
#         done
#     done
# done

# CUDA_VISIBLE_DEVICES=5 python train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/hiv/" --task "hiv" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --disable_lora --train_on_inputs --learning_rate 2e-4 --cutoff_len 2048
# CUDA_VISIBLE_DEVICES=5 python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/galactica-125M/hiv/checkpoint-10280/" --task "hiv" --output_dir "./moleculenet_predictions/galactica-125M/" --disable_lora

# for lr in 2e-4
# do
#     for n_shot in 2 4
#     do
#         for len in 2048
#         do
#             # CUDA_VISIBLE_DEVICES=0 python train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/clintox-R-$lr-$n_shot-$len/" --task "clintox" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk
#             CUDA_VISIBLE_DEVICES=0 python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/galactica-125M/clintox-R-$lr-$n_shot-$len/checkpoint-740/" --task "clintox" --output_dir "./moleculenet_predictions/galactica-125M-R-$lr-$n_shot-$len/" --retrieval --disable_lora --n_shot $n_shot --reverse --random_walk --batch_infer
#         done
#     done
# done

# # CUDA_VISIBLE_DEVICES=0 python train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M/clintox/" --task "clintox" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --disable_lora --train_on_inputs --learning_rate 2e-4 --cutoff_len 2048
# CUDA_VISIBLE_DEVICES=0 python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/galactica-125M/clintox/checkpoint-740/" --task "clintox" --output_dir "./moleculenet_predictions/galactica-125M/" --disable_lora

# =========Mistral 7B===========

# for lr in 2e-4
# do
#     for n_shot in 2
#     do
#         for len in 3048
#         do
#             accelerate launch train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "mistralai/Mistral-7B-Instruct-v0.2" --output_dir "./ckp/Mistral-7B/bace-R-$lr-$n_shot-$len/" --task "bace" --micro_batch_size 1 --batch_size 32 --num_epochs 10 --retrieval --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk
#             python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/Mistral-7B/bace-R-$lr-$n_shot-$len/checkpoint-370/" --task "bace" --output_dir "./moleculenet_predictions/Mistral-7B-R-$lr-$n_shot-$len/" --retrieval --n_shot $n_shot --reverse --random_walk --max_new_tokens 10
#             python naive_test.py --raw_folder "./data/MoleculeNet/" --target_folder "moleculenet_predictions/" --model "Mistral-7B-R-$lr-$n_shot-$len" --ckp 370 --task bace
#         done
#     done
# done

# accelerate launch train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "mistralai/Mistral-7B-Instruct-v0.2" --output_dir "./ckp/Mistral-7B/bace/" --task "bace" --micro_batch_size 1 --batch_size 32 --num_epochs 10 --train_on_inputs --learning_rate 2e-4 --cutoff_len 1024
# python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/Mistral-7B/bace/checkpoint-370/" --task "bace" --output_dir "./moleculenet_predictions/Mistral-7B/" --max_new_tokens 10
# python naive_test.py --raw_folder "./data/MoleculeNet/" --target_folder "moleculenet_predictions/" --model "Mistral-7B" --ckp 370 --task bace



# for lr in 2e-4
# do
#     for n_shot in 2 4
#     do
#         for len in 1024
#         do
#             # accelerate launch train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "mistralai/Mistral-7B-Instruct-v0.2" --output_dir "./ckp/Mistral-7B/bbbp-R-$lr-$n_shot-$len/" --task "bbbp" --micro_batch_size 1 --batch_size 32 --num_epochs 10 --retrieval --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk --int8 --fp16
#             # python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/Mistral-7B/bbbp-R-$lr-$n_shot-$len/checkpoint-510/" --task "bbbp" --output_dir "./moleculenet_predictions/Mistral-7B-R-$lr-$n_shot-$len/" --retrieval --n_shot $n_shot --reverse --random_walk --max_new_tokens 10 --int8 --fp16
#             # python naive_test.py --raw_folder "./data/MoleculeNet/" --target_folder "moleculenet_predictions/" --model "Mistral-7B-R-$lr-$n_shot-$len" --ckp 510 --task bbbp
#         done
#     done
# done

# accelerate launch train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "mistralai/Mistral-7B-Instruct-v0.2" --output_dir "./ckp/Mistral-7B/bbbp/" --task "bbbp" --micro_batch_size 1 --batch_size 32 --num_epochs 10 --train_on_inputs --learning_rate 2e-4 --cutoff_len 1024
# python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/Mistral-7B/bbbp/checkpoint-510/" --task "bbbp" --output_dir "./moleculenet_predictions/Mistral-7B/" --max_new_tokens 10
# python naive_test.py --raw_folder "./data/MoleculeNet/" --target_folder "moleculenet_predictions/" --model "Mistral-7B" --ckp 510 --task bbbp


# for lr in 2e-4
# do
#     for n_shot in 2 4
#     do
#         for len in 512
#         do
#             # accelerate launch train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "mistralai/Mistral-7B-Instruct-v0.2" --output_dir "./ckp/Mistral-7B/sider-R-$lr-$n_shot-$len/" --task "sider" --micro_batch_size 1 --batch_size 32 --num_epochs 10 --retrieval --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk 
#             # python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/Mistral-7B/sider-R-$lr-$n_shot-$len/checkpoint-9620/" --task "sider" --output_dir "./moleculenet_predictions/Mistral-7B-R-$lr-$n_shot-$len/" --retrieval --n_shot $n_shot --reverse --random_walk --max_new_tokens 10 
#             # python naive_test.py --raw_folder "./data/MoleculeNet/" --target_folder "moleculenet_predictions/" --model "Mistral-7B-R-$lr-$n_shot-$len" --ckp 9620 --task sider
#         done
#     done
# done

# accelerate launch train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "mistralai/Mistral-7B-Instruct-v0.2" --output_dir "./ckp/Mistral-7B/sider/" --task "sider" --micro_batch_size 1 --batch_size 32 --num_epochs 10 --train_on_inputs --learning_rate 2e-4 --cutoff_len 512
# python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/Mistral-7B/sider/checkpoint-9620/" --task "sider" --output_dir "./moleculenet_predictions/Mistral-7B/" --max_new_tokens 10 
# python naive_test.py --raw_folder "./data/MoleculeNet/" --target_folder "moleculenet_predictions/" --model "Mistral-7B" --ckp 9620 --task sider


# for lr in 2e-4
# do
#     for n_shot in 2 4
#     do
#         for len in 512
#         do
#             # accelerate launch train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "mistralai/Mistral-7B-Instruct-v0.2" --output_dir "./ckp/Mistral-7B/tox21-R-$lr-$n_shot-$len/" --task "tox21" --micro_batch_size 1 --batch_size 32 --num_epochs 10 --retrieval --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk 
#             # python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/Mistral-7B/tox21-R-$lr-$n_shot-$len/checkpoint-19420/" --task "tox21" --output_dir "./moleculenet_predictions/Mistral-7B-R-$lr-$n_shot-$len/" --retrieval --n_shot $n_shot --reverse --random_walk --max_new_tokens 10 
#             # python naive_test.py --raw_folder "./data/MoleculeNet/" --target_folder "moleculenet_predictions/" --model "Mistral-7B-R-$lr-$n_shot-$len" --ckp 19420 --task tox21
#         done
#     done
# done

# accelerate launch train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "mistralai/Mistral-7B-Instruct-v0.2" --output_dir "./ckp/Mistral-7B/tox21/" --task "tox21" --micro_batch_size 1 --batch_size 32 --num_epochs 10 --train_on_inputs --learning_rate 2e-4 --cutoff_len 512
# python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/Mistral-7B/tox21/checkpoint-19420/" --task "tox21" --output_dir "./moleculenet_predictions/Mistral-7B/" --max_new_tokens 10 
# python naive_test.py --raw_folder "./data/MoleculeNet/" --target_folder "moleculenet_predictions/" --model "Mistral-7B" --ckp 19420 --task tox21


# for lr in 2e-4
# do
#     for n_shot in 2 4
#     do
#         for len in 512
#         do
#             # accelerate launch train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "mistralai/Mistral-7B-Instruct-v0.2" --output_dir "./ckp/Mistral-7B/hiv-R-$lr-$n_shot-$len/" --task "hiv" --micro_batch_size 1 --batch_size 32 --num_epochs 10 --retrieval --n_shot $n_shot --train_on_inputs --learning_rate $lr --cutoff_len $len --reverse --random_walk 
#             # python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/Mistral-7B/hiv-R-$lr-$n_shot-$len/checkpoint-10280/" --task "hiv" --output_dir "./moleculenet_predictions/Mistral-7B-R-$lr-$n_shot-$len/" --retrieval --n_shot $n_shot --reverse --random_walk --max_new_tokens 10 
#             # python naive_test.py --raw_folder "./data/MoleculeNet/" --target_folder "moleculenet_predictions/" --model "Mistral-7B-R-$lr-$n_shot-$len" --ckp 10280 --task hiv
#         done
#     done
# done

# accelerate launch train_ppo.py --data_folder "./data/MoleculeNet/" --base_model "mistralai/Mistral-7B-Instruct-v0.2" --output_dir "./ckp/Mistral-7B/hiv/" --task "hiv" --micro_batch_size 1 --batch_size 32 --num_epochs 10 --train_on_inputs --learning_rate 2e-4 --cutoff_len 512
# python test.py --data_folder "./data/MoleculeNet/" --adapter_path "./ckp/Mistral-7B/hiv/checkpoint-10280/" --task "hiv" --output_dir "./moleculenet_predictions/Mistral-7B/" --max_new_tokens 10 
# python naive_test.py --raw_folder "./data/MoleculeNet/" --target_folder "moleculenet_predictions/" --model "Mistral-7B" --ckp 10280 --task hiv


# +++++++++++Galactica 125M++++++++++++


# for seed in 42    # bucket sampling baseline
# do
#     for lr in 2e-4
#     do
#         for n_shot in 1 2
#         do
#             for len in 512 1024 1536 2048
#             do
#                 CUDA_VISIBLE_DEVICES=7 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M-B/mol2cap-$lr-$n_shot-$len-$seed/" --task "mol2cap" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --learning_rate $lr --cutoff_len $len --reverse --bucket_sampling --seed $seed
#                 CUDA_VISIBLE_DEVICES=7 python test.py --adapter_path "./ckp/galactica-125M-B/mol2cap-$lr-$n_shot-$len-$seed/checkpoint-8000/" --task "mol2cap" --output_dir "./rebuttal_predictions/galactica-125M-B-$lr-$n_shot-$len-$seed/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --bucket_sampling --seed $seed
            
#                 CUDA_VISIBLE_DEVICES=7 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M-B/cap2mol-$lr-$n_shot-$len-$seed/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --learning_rate $lr --cutoff_len $len --reverse --bucket_sampling --seed $seed
#                 CUDA_VISIBLE_DEVICES=7 python test.py --adapter_path "./ckp/galactica-125M-B/cap2mol-$lr-$n_shot-$len-$seed/checkpoint-8000/" --task "cap2mol" --output_dir "./rebuttal_predictions/galactica-125M-B-$lr-$n_shot-$len-$seed/" --retrieval --disable_lora --batch_infer --n_shot $n_shot --reverse --bucket_sampling --seed $seed
#             done
#         done
#     done
# done

# for seed in 1024 179 654 1234
# do
#     for lr in 2e-4
#     do
#         for n_shot in 2
#         do
#             for len in 1024
#             do
#                 for skip in 5
#                 do
#                     CUDA_VISIBLE_DEVICES=5 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M-R/mol2cap-$lr-$n_shot-$len-$seed-$skip/" --task "mol2cap" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --learning_rate $lr --cutoff_len $len --reverse --random_walk --seed $seed --train_on_inputs --max_skip_prob $skip
#                     CUDA_VISIBLE_DEVICES=5 python test.py --adapter_path "./ckp/galactica-125M-R/mol2cap-$lr-$n_shot-$len-$seed-$skip/checkpoint-8000/" --task "mol2cap" --output_dir "./rebuttal_predictions/galactica-125M-R-$lr-$n_shot-$len-$seed-$skip/" --retrieval --disable_lora --n_shot $n_shot --reverse --random_walk --seed $seed --max_skip_prob $skip
                
#                     # CUDA_VISIBLE_DEVICES=7 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M-R/cap2mol-$lr-$n_shot-$len-$seed/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --learning_rate $lr --cutoff_len $len --reverse --random_walk --seed $seed --train_on_inputs
#                     # CUDA_VISIBLE_DEVICES=7 python test.py --adapter_path "./ckp/galactica-125M-R/cap2mol-$lr-$n_shot-$len-$seed/checkpoint-8000/" --task "cap2mol" --output_dir "./rebuttal_predictions/galactica-125M-R-$lr-$n_shot-$len-$seed/" --retrieval --disable_lora --n_shot $n_shot --reverse --random_walk --seed $seed
#                 done
#             done
#         done
#     done
# done

# for seed in 396 117 128
# do
#     for lr in 2e-4
#     do
#         for n_shot in 4
#         do
#             for len in 2048
#             do
#                 for skip in 1 3 5 15 20 50
#                 do
#                     # CUDA_VISIBLE_DEVICES=7 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M-R/mol2cap-$lr-$n_shot-$len-$seed/" --task "mol2cap" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --learning_rate $lr --cutoff_len $len --reverse --random_walk --seed $seed --train_on_inputs
#                     # CUDA_VISIBLE_DEVICES=7 python test.py --adapter_path "./ckp/galactica-125M-R/mol2cap-$lr-$n_shot-$len-$seed/checkpoint-8000/" --task "mol2cap" --output_dir "./rebuttal_predictions/galactica-125M-R-$lr-$n_shot-$len-$seed/" --retrieval --disable_lora --n_shot $n_shot --reverse --random_walk --seed $seed
                
#                     CUDA_VISIBLE_DEVICES=7 python train_ppo.py --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M-R/cap2mol-$lr-$n_shot-$len-$seed-$skip/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --learning_rate $lr --cutoff_len $len --reverse --random_walk --seed $seed --train_on_inputs --max_skip_prob $skip
#                     CUDA_VISIBLE_DEVICES=7 python test.py --adapter_path "./ckp/galactica-125M-R/cap2mol-$lr-$n_shot-$len-$seed-$skip/checkpoint-8000/" --task "cap2mol" --output_dir "./rebuttal_predictions/galactica-125M-R-$lr-$n_shot-$len-$seed-$skip/" --retrieval --disable_lora --n_shot $n_shot --reverse --random_walk --seed $seed --max_skip_prob $skip
#                 done
#             done
#         done
#     done
# done


# for lr in 2e-4
# do
#     for n_shot in 2
#     do
#         for len in 1024
#         do
#             CUDA_VISIBLE_DEVICES=6 python train_ppo.py --data_folder "./data/PubChem324k/molcap/raw/" --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M-Pub/cap2mol-$lr-$n_shot-$len/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --learning_rate $lr --cutoff_len $len --reverse --random_walk
#             CUDA_VISIBLE_DEVICES=6 python test.py --data_folder "./data/PubChem324k/molcap/raw/" --adapter_path "./ckp/galactica-125M-Pub/cap2mol-$lr-$n_shot-$len/checkpoint-3750/" --task "cap2mol" --output_dir "./rebuttal_predictions/galactica-125M-P-$lr-$n_shot-$len/" --batch_infer --retrieval --disable_lora --n_shot $n_shot --reverse --random_walk
#         done
#     done
# done

# for lr in 2e-4
# do
#     for n_shot in 4
#     do
#         for len in 2048
#         do
#             CUDA_VISIBLE_DEVICES=6 python train_ppo.py --data_folder "./data/PubChem324k/molcap/raw/" --base_model "facebook/galactica-125m" --output_dir "./ckp/galactica-125M-Pub/cap2mol-$lr-$n_shot-$len/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --retrieval --disable_lora --n_shot $n_shot --learning_rate $lr --cutoff_len $len --reverse --random_walk
#             CUDA_VISIBLE_DEVICES=6 python test.py --data_folder "./data/PubChem324k/molcap/raw/" --adapter_path "./ckp/galactica-125M-Pub/cap2mol-$lr-$n_shot-$len/checkpoint-3750/" --task "cap2mol" --output_dir "./rebuttal_predictions/galactica-125M-P-$lr-$n_shot-$len/" --batch_infer --retrieval --disable_lora --n_shot $n_shot --reverse --random_walk
#         done
#     done
# done

# for seed in 396
# do
#     for lr in 3e-4
#     do
#         for n_shot in 2
#         do
#             for len in 2048
#             do
#                 # CUDA_VISIBLE_DEVICES=6 python train_ppo.py --base_model "mistralai/Mistral-7B-Instruct-v0.2" --output_dir "./ckp/mistral-7B/cap2mol-$lr-$n_shot-$len-$seed/" --task "cap2mol" --micro_batch_size 2 --batch_size 32 --num_epochs 10 --retrieval --n_shot $n_shot --learning_rate $lr --cutoff_len $len --reverse --random_walk --seed $seed --train_on_inputs --int8 --fp16
#                 # CUDA_VISIBLE_DEVICES=6 python test.py --adapter_path "./ckp/mistral-7B/cap2mol-$lr-$n_shot-$len-$seed/checkpoint-8000/" --task "cap2mol" --output_dir "./rebuttal_predictions/mistral-7B-R-$lr-$n_shot-$len-$seed/" --retrieval --n_shot $n_shot --reverse --random_walk --seed $seed
            
#                 # CUDA_VISIBLE_DEVICES=5 python train_ppo.py --base_model "facebook/galactica-1.3b" --output_dir "./ckp/galactica-1B-R/cap2mol-$lr-$n_shot-$len-$seed/" --task "cap2mol" --micro_batch_size 1 --batch_size 32 --num_epochs 10 --retrieval --n_shot $n_shot --learning_rate $lr --cutoff_len $len --reverse --random_walk --seed $seed --train_on_inputs
#                 # CUDA_VISIBLE_DEVICES=5 python test.py --adapter_path "./ckp/galactica-1B-R/cap2mol-$lr-$n_shot-$len-$seed/checkpoint-8000/" --task "cap2mol" --output_dir "./rebuttal_predictions/galactica-1B-R-$lr-$n_shot-$len-$seed/" --retrieval --n_shot $n_shot --reverse --random_walk --seed $seed
            
#                 CUDA_VISIBLE_DEVICES=5 python train_ppo.py --base_model "./ckp/OMG-125M-xlarge" --output_dir "./ckp/OMG/" --task "cap2mol" --micro_batch_size 4 --batch_size 32 --num_epochs 10 --learning_rate $lr --cutoff_len $len --seed $seed --train_on_inputs --add_special_token
#                 CUDA_VISIBLE_DEVICES=5 python test.py --base_model "facebook/galactica-125m" --adapter_path "./ckp/OMG/checkpoint-8000/" --task "cap2mol" --output_dir "./predictions/temp/" --seed 396 --add_special_token
            
#                 # CUDA_VISIBLE_DEVICES=6 python train_ppo.py --base_model "meta-llama/Llama-3.2-3B-Instruct" --output_dir "./ckp/llama3-3B-R/cap2mol-$lr-$n_shot-$len-$seed/" --task "cap2mol" --micro_batch_size 1 --batch_size 32 --num_epochs 10 --retrieval --n_shot $n_shot --learning_rate $lr --cutoff_len $len --reverse --random_walk --seed $seed --train_on_inputs --int8 --fp16 --add_eos "<|end_of_text|>"
#                 # CUDA_VISIBLE_DEVICES=6 python test.py --base_model "meta-llama/Llama-3.2-3B-Instruct" --adapter_path "./ckp/llama3-3B-R/cap2mol-$lr-$n_shot-$len-$seed/checkpoint-8000/" --task "cap2mol" --output_dir "./rebuttal_predictions/llama3-3B-R-$lr-$n_shot-$len-$seed/" --retrieval --n_shot $n_shot --reverse --random_walk --seed $seed

#                 # CUDA_VISIBLE_DEVICES=6 python train_ppo.py --base_model "meta-llama/Meta-Llama-3-8B-Instruct" --output_dir "./ckp/llama3-8B-R/cap2mol-$lr-$n_shot-$len-$seed/" --task "cap2mol" --micro_batch_size 1 --batch_size 32 --num_epochs 10 --retrieval --n_shot $n_shot --learning_rate $lr --cutoff_len $len --reverse --random_walk --seed $seed --train_on_inputs --int8 --fp16 --add_eos "<|end_of_text|>"
#                 # CUDA_VISIBLE_DEVICES=6 python test.py --base_model "meta-llama/Meta-Llama-3-8B-Instruct" --adapter_path "./ckp/llama3-8B-R/cap2mol-$lr-$n_shot-$len-$seed/checkpoint-8000/" --task "cap2mol" --output_dir "./rebuttal_predictions/llama3-8B-R-$lr-$n_shot-$len-$seed/" --retrieval --n_shot $n_shot --reverse --random_walk --seed $seed
            
#             done
#         done
#     done
# done


