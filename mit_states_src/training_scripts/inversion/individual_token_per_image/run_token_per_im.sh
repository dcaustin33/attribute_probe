python3 train_weird_texture.py -train  --data_dir human_4_leg/stitched_realistic --output_dir human_4_leg/outputs_texture_sti_real --name human_4_leg/outputs_texture_sti_real --steps 3;
python3 train_with.py -train --data_dir human_4_leg/stitched_realistic --output_dir human_4_leg/outputs_texture_sti_real --name human_4_leg/outputs_texture_sti_real --steps 3;
python3 train_stitched_concept.py -train --data_dir human_4_leg/stitched_realistic --output_dir human_4_leg/outputs_texture_sti_real --name human_4_leg/outputs_texture_sti_real --steps 3;

python3 train_weird_texture.py -train --data_dir mandarin_fish/tshirt_stitches --output_dir mandarin_fish/outputs_texture_sti_real --name mandarin_fish/outputs_texture_sti_real --steps 3;
python3 train_with.py -train --data_dir mandarin_fish/tshirt_stitches --output_dir mandarin_fish/outputs_texture_sti_real --name mandarin_fish/outputs_texture_sti_real --steps 3;
python3 train_stitched_concept.py -train --data_dir mandarin_fish/tshirt_stitches --output_dir mandarin_fish/outputs_texture_sti_real --name mandarin_fish/outputs_texture_sti_real --steps 3;

python3 train_weird_texture.py -train --data_dir worms/stitched_realistic_worms --output_dir worms/outputs_texture_sti_real --name worms/outputs_texture_sti_real --steps 3;
python3 train_with.py -train --data_dir worms/stitched_realistic_worms --output_dir worms/outputs_texture_sti_real --name worms/outputs_texture_sti_real --steps 3;
python3 train_stitched_concept.py -train --data_dir worms/stitched_realistic_worms --output_dir worms/outputs_texture_sti_real --name worms/outputs_texture_sti_real --steps 3;



# sudo shutdown -h;