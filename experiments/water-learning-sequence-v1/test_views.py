"""Transport contract regressions; synthetic arrays make no physics claim."""
import copy
import io
import json
from pathlib import Path
import unittest
import numpy as np
import data
import scenes
from policy import LAB, canonical, sha


def fixture():
    config=json.loads((LAB/'experiments/sealed-water-v1/configs/water.json').read_text())
    reservation=scenes.reserve(config);scene=next(row for row in reservation['scenes'] if row['split']=='train')
    arrays={f'p0000_{name}':np.zeros((1,16,16,16),dtype='<f8') for name in data.FIELDS}
    inventory=data.inventory(arrays);keys=list(arrays);payloads=[];shards=[]
    for selected in (keys[:2],keys[2:]):
        buffer=io.BytesIO();np.savez(buffer,**{key:arrays[key] for key in selected});raw=buffer.getvalue();payloads.append(raw)
        shards.append({'sha256':sha(raw),'bytes':len(raw),'keys':selected})
    parent={'source':'test-source','scene':scene,'entry':{'trajectory_id':'test-source--'+scene['trajectory_id'],'split':'train','dt_s':1/60},
        'counts_xyz':[16,16,16],'cell_size_m':1/scene['cells_per_metre'],'density_kg_m3':1000.0,'capture_root':'/not-a-physical-fixture','capture_manifest_sha256':'1'*64,
        'frames':[{'index':15,'sha256':{name:sha(arrays['p0000_'+name][0].tobytes()) for name in data.FIELDS}}]}
    selected={'schema':'rheon-water-learning-reservation-v1','sources':{'test-source':{'reservation':reservation,'capture_manifest_sha256':'1'*64,'packet_manifest_sha256':'2'*64}},
        'selection':[{'source':'test-source','trajectory_id':scene['trajectory_id'],'frames':[15]}]}
    manifest={'schema':'rheon-water-learning-view-v1','reservation':selected,'reservation_sha256':sha(canonical(selected)),'shards':shards,'parents':[parent],'arrays':inventory,'population':'train_validation'}
    return tuple(payloads),manifest


class ViewTests(unittest.TestCase):
    def test_round_trip_preserves_full_cubic_grid_and_preceding_pressure(self):
        payloads,manifest=fixture();_,parents=data.load(payloads,manifest)
        self.assertEqual(parents[0]['arrays']['base'].shape,(1,16,16,16))
        self.assertEqual(parents[0]['frames'][0]['index'],15)
        self.assertEqual(parents[0]['entry']['trajectory_id'],'test-source--water-train-0000')
        self.assertEqual(parents[0]['cell_size_m'],1/8)

    def test_changed_shard_is_rejected_before_numeric_loading(self):
        payloads,manifest=fixture();changed=(payloads[0][:-1]+b'x',payloads[1])
        with self.assertRaisesRegex(ValueError,'view_shard_hash'):data.load(changed,manifest)

    def test_wrong_split_spacing_and_source_are_rejected(self):
        payloads,manifest=fixture()
        for name,mutate,code in [
            ('split',lambda m:m.update(population='reserved_test'),'view_split'),
            ('spacing',lambda m:m['parents'][0].update(cell_size_m=.125*2),'view_cubic'),
            ('source',lambda m:m['parents'][0]['entry'].update(trajectory_id='invented-id'),'view_lineage')]:
            value=copy.deepcopy(manifest);mutate(value)
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,code):data.load(payloads,value)

    def test_training_order_cannot_sample_validation(self):
        import train
        config=json.loads((Path(__file__).parent/'configs/fit-one.json').read_text())
        config['training']['updates']=50
        parents=[{'entry':{'split':'train'},'frames':[{}]},{'entry':{'split':'validation'},'frames':[{},{}]}]
        order=train.order(parents,config)
        self.assertEqual(order.shape,(50,1,2))
        self.assertTrue((order==0).all())


if __name__=='__main__':unittest.main()
