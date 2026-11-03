"""Regression: own-history training must feed exactly the deployed state encoding."""
import argparse
import numpy as np
import torch
from study_support import ROOT,BASE,read
from bundle_model_v5 import BundleNet
from temporal_features_v2 import FIELD_NAMES,packed_state,auxiliary,unpack_outputs

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--legacy',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    bank=read(BASE/'temporal-data-01/bank.json')
    row=next(r for r in bank['trajectories']if r['split']=='train'and r['counts_xyz']==[16,16,16])
    receipt_path=ROOT/row['receipt'];receipt=read(receipt_path)
    with np.load(receipt_path.parent/receipt['states'][8]['path'],allow_pickle=False)as source:
        fields={name:source[name].copy()for name in FIELD_NAMES}
    norm=read(BASE/'temporal-training-view-02/normalization.json');counts=tuple(receipt['scene']['counts'])
    torch.manual_seed(2026101921);network=BundleNet(2).eval()
    with torch.no_grad():
        current=packed_state(fields,counts,norm['scales'])
        predicted=network(current,auxiliary(receipt['scene'],8,2,norm['scales']))
        deployed=packed_state(unpack_outputs(predicted[:,-1:],counts,norm['scales'])[0],counts,norm['scales'])
        if args.legacy:training=predicted[:,-1]
        else:
            from temporal_feedback import next_input
            training=next_input(predicted,counts,norm['scales'])
        difference=float((training-deployed).abs().max())
        print('Maximum own-history input discrepancy',difference,flush=True)
        torch.testing.assert_close(training,deployed,rtol=1e-6,atol=1e-6)
    print('Training feedback matches deployed physical reconstruction and encoding',flush=True)
