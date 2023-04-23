python3 train_style.py -train --data_dir coral/beard_pics --output_dir coral/outputs_beard_style --name coral/outputs_beard_style_model --steps 3000;
python3 train_style.py -train --data_dir coral/solo_pics --output_dir coral/outputs_solo_style --name coral/outputs_solo_style_model --steps 3000;
python3 train_weird_texture.py -train --data_dir coral/beard_pics --output_dir coral/outputs_beard_texture --name coral/outputs_beard_texture_model --steps 3000;
python3 train_weird_texture.py -train --data_dir coral/solo_pics --output_dir coral/outputs_solo_texture --name coral/outputs_solo_texture_model --steps 3000;

python3 train_style.py -train --data_dir barnacles/beard_pics --output_dir barnacles/outputs_beard_style --name barnacles/outputs_beard_style_model --steps 3000;
python3 train_style.py -train --data_dir barnacles/solo_pics --output_dir barnacles/outputs_solo_style --name barnacles/outputs_solo_style_model --steps 3000;
python3 train_weird_texture.py -train --data_dir barnacles/beard_pics --output_dir barnacles/outputs_beard_texture --name barnacles/outputs_beard_texture_model --steps 3000;
python3 train_weird_texture.py -train --data_dir barnacles/solo_pics --output_dir barnacles/outputs_solo_texture --name barnacles/outputs_solo_texture_model --steps 3000;

python3 train_style.py -train --data_dir lichen/beard_pics --output_dir lichen/outputs_beard_style --name lichen/outputs_beard_style_model --steps 3000;
python3 train_style.py -train --data_dir lichen/solo_pics --output_dir lichen/outputs_solo_style --name lichen/outputs_solo_style_model --steps 3000;
python3 train_weird_texture.py -train --data_dir lichen/beard_pics --output_dir lichen/outputs_beard_texture --name lichen/outputs_beard_texture_model --steps 3000;
python3 train_weird_texture.py -train --data_dir lichen/solo_pics --output_dir lichen/outputs_solo_texture --name lichen/outputs_solo_texture_model --steps 3000;


sudo shutdown -h;