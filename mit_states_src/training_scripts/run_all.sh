#cd inversion/strange_colors;
#python3 neon_green.py -train --data_dir neon_green_pics_4 --output_dir output/neon_green_pics_4 --name neon_green_pics_4_concept;
python3 neon_green.py -train --data_dir neon_green_spheres --output_dir output/neon_green_pics_spheres --name neon_green_pics_spheres_concept;
python3 neon_green.py -train --data_dir magenta_4 --output_dir output/magenta_pics_4 --name magenta_pics_4_concept;
#sudo shutdown -h;