cd /home/anthony/sls/abau/nmt-saliency

for lang in 'fr' 'ar' 'ru' 'zh'
do
    for n in 1 2 3
    do
    echo "/home/anthony/sls/models/en-${lang}-2m-${n}-model_final.t7\n/home/anthony/sls/dicts/en-${lang}-2m-${n}.src.dict\n/home/anthony/sls/descriptions/en-${lang}-${n}.desc.t7" > model_list.txt
    th all-saliencies.lua \
        -model_list model_list.txt \
        -src_file /home/anthony/sls/data/sample/en.tok \
        -out_file /home/anthony/sls/src/results/saliency-${lang}-${n}.json \
        -max_len 50
    done
done
