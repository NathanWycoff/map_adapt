start=`date +%s`
python python/clean_dir_make_args.py small
#parallel --jobs 20 --colsep ' ' --will-cite -a sim_args.txt python python/ada_sim.py
#parallel --jobs 5 --colsep ' ' --will-cite -a sim_args.txt python python/ada_sim.py
#parallel --nice 5 --jobs 5 --colsep ' ' --will-cite -a sim_args.txt python python/ada_sim.py
parallel --nice 5 --jobs 10 --colsep ' ' --will-cite -a sim_args.txt python python/ada_sim.py
#echo Only 1 thread.
#parallel --nice 5 --jobs 1 --colsep ' ' --will-cite -a sim_args.txt python python/ada_sim.py
python python/plot_ada_sim2.py 
end=`date +%s`
runtime=$((end-start))
echo $runtime
