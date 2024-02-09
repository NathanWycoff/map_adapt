start=`date +%s`
python python/clean_dir_make_args.py hcr
#parallel --jobs 20 --colsep ' ' --will-cite -a sim_args.txt python python/ada_sim.py
#parallel --jobs 5 --colsep ' ' --will-cite -a sim_args.txt python python/ada_sim.py
parallel --nice 5 --jobs 10 --colsep ' ' --will-cite -a hcr_zinb_args.txt python python/neo_hcr.py
#parallel --nice 5 --jobs 2 --colsep ' ' --will-cite -a hcr_comp_args.txt python python/hcr_competitors.py
python python/plot_hcr.py 
end=`date +%s`
runtime=$((end-start))
echo $runtime
