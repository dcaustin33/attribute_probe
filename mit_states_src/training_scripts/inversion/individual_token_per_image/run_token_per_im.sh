python3 token_per_image.py -train --data_dir ng_4/neon_green_pics_4 --output_dir ng_4/output/neon_green_pics_4 --name ng_4/neon_green_pics_4_concept_throughout;
python3 token_per_image.py -train --data_dir ng_8/neon_green_pics_8 --output_dir ng_8/output/neon_green_pics_8 --name ng_8/neon_green_pics_8_concept_throughout;
python3 token_per_image.py -train --data_dir ng_20/neon_green_pics_20 --output_dir ng_20/output/neon_green_pics_20 --name ng_20/neon_green_pics_20_concept_throughout;

python3 token_per_image.py -train --data_dir mg_4/magenta_pics_4 --output_dir mg_4/output/magenta_pics_4 --name mg_4/magenta_pics_4_concept_throughout;
python3 token_per_image.py -train --data_dir mg_8/magenta_pics_8 --output_dir mg_8/output/magenta_pics_8 --name mg_8/magenta_pics_8_concept_throughout;
python3 token_per_image.py -train --data_dir mg_20/magenta_pics_20 --output_dir mg_20/output/magenta_pics_20 --name mg_20/magenta_pics_20_concept_throughout;

python3 token_per_image.py -train --data_dir stripes/stripes_10 --output_dir stripes/output/stripes_10 --name stripes/stripes_10_concept_throughout;
python3 token_per_image.py -train --data_dir stripes/stripes_20 --output_dir stripes/output/stripes_20 --name stripes/stripes_20_concept_throughout;

sudo shutdown -h;
