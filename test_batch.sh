for i in {101..110}
do
    python examples/test_ngp_nerf.py --data_root /home/woongohcho/data/kubric360_v2 --scene $i --checkpoint checkpoint/${i}_model_40000.pth
done
