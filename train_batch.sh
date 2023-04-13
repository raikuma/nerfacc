for i in {101..110}
do
    if [ `cat logs/$i.log | wc -l` -eq 18 ]
    then
        echo "$i is complete"
    else
        echo "$i is incomplete"
        python examples/train_ngp_nerf.py --data_root /home/woongohcho/data/kubric360_v2 --scene $i > logs/$i.log
    fi
done
