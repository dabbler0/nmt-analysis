cd /home/anthony/sls/abau/nmt-shared-information
for LANG in en ar fr es ru zh
do
    for SET in 1 2 3
    do
        th describe.lua \
            -model "/home/anthony/sls/models/en-${LANG}-2m-${SET}-model_final.t7" \
            -src_dict "/home/anthony/sls/dicts/en-${LANG}-2m-${SET}.src.dict" \
            -targ_dict "/home/anthony/sls/dicts/en-${LANG}-2m-${SET}.targ.dict" \
            -src_file "/home/anthony/sls/data/testsets/tokenized-test/en.tok" \
            -output_file "/home/anthony/sls/descriptions/en-${LANG}-${SET}.desc.t7"
    done
done
