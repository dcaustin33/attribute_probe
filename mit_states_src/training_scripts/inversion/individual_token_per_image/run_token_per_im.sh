# trying different training steps for the run
python3 train_weird_texture.py -train  --data_dir human_4_leg/pics --output_dir human_4_leg/outputs1000 --name human_4_leg_concept1000/ --steps 1000;
python3 train_weird_texture.py -train  --data_dir human_4_leg/pics --output_dir human_4_leg/outputs3000 --name human_4_leg_concept3000/ --steps 3000;

# try neon green with stitching just change the script to train_weird_color.py
python3 train_weird_color.py -train --data_dir ng_stitched/pics --output_dir ng_stitched/outputs_color --name ng_stitched/concept_color;

# try neon green with no stitching and the name color
python3 train_weird_color.py -train --data_dir ng_only/pics --output_dir ng_only/outputs --name ng_only/concept;

python3 train_weird_texture.py -train  --data_dir human_4_leg/pics_dancers --output_dir human_4_leg/outputsdancers --name human_4_leg_conceptdancers --steps 3000;
python3 train_weird_texture.py -train  --data_dir human_4_leg/stiched_photos --output_dir human_4_leg/outputsstiched --name human_4_leg_conceptstiched --steps 3000;

# try mandarin fish
python3 train_weird_texture.py -train  --data_dir mandarin_fish/stiched_pics --output_dir mandarin_fish/outputs_stiched --name mandarin_fish/concept_stiched --steps 3000;
python3 train_weird_color.py -train --data_dir mandarin_fish/stiched_pics --output_dir mandarin_fish/outputs_stiched_color --name mandarin_fish/concept_stiched_color --steps 3000;

python3 train_weird_texture.py -train  --data_dir mandarin_fish/pics --output_dir mandarin_fish/outputs_regular --name mandarin_fish/concept_regular --steps 3000;
python3 train_weird_color.py -train --data_dir mandarin_fish/pics --output_dir mandarin_fish/outputs_regular_color --name mandarin_fish/concept_regular_color --steps 3000;

# try the jelly fish
python3 train_weird_texture.py -train  --data_dir jellyfish/jellyfish_pics2 --output_dir jelly_fish/outputs_regular --name jelly_fish/concept_regular --steps 3000;
python3 train_weird_color.py -train --data_dir jellyfish/jellyfish_pics2 --output_dir jelly_fish/outputs_regular_color --name jelly_fish/concept_regular_color --steps 3000;

sudo shutdown -h;