"""Pure contract fixtures: no model import, forward, scientific native invocation."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("gn_admission_fixture",HERE/"admission.py")
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def receipt_fixture():
    # Metadata-only seed; every mutation is disposable and never an admitted
    # candidate. No weights, source fields or scientific binary are executed.
    return a.strict_json((a.LAB/"runs/global-pressure-groupnorm-v1/restore-stage/artifacts/receipt.json").read_bytes())


def archive_fixture(values=None):
    values=np.array([1.,-2.],dtype="<f8") if values is None else values
    blob=io.BytesIO();np.savez(blob,field=values)
    entry={"dtype":values.dtype.str,"shape":list(values.shape),"bytes":values.nbytes,
           "sha256":a.digest(values.tobytes())}
    return blob.getvalue(),{"field":entry}


def request_fixture():
    return {"schema":"rheon-global-pressure-teacher-case-v1","counts_xyz":[16]*3,
            "spacing_m":[.25,.4,.7],"density_kg_m3":1.25,"dt_s":1/60,
            "original_rhs":[0.]*4096,"reference_pressure":[0.]*4096,
            "previous_pressure":[0.]+[1.]*4095,"candidate_pressure":[0.]+[2.]*4095,
            "candidate_error":None,"pre_velocity":[[0.]*4352]*3}


def quality_fixture():
    return {"qualified":True,"failures":[],"residual_l2":0.,"residual_max":0.,"rhs_l2":0.,
            "relative_residual":None,"predicted_divergence_max_s_inv":0.,
            "actual_divergence_max_s_inv":0.,"actual_divergence_rms_s_inv":0.,
            "net_flux_m3_s":0.,"wall_normal_flux_exact_zero":True,"courant":0.,"gauge_cell":0.}


def solve_fixture(request,limit,guess,qualified=True):
    q=quality_fixture()
    return {"qualified":qualified,"solver_error":None if qualified else "iteration limit",
            "iterations":0 if qualified else limit,"iteration_limit":limit,
            "initial_guess_sha256":a.digest(bytes(4096*8)) if guess is None else a.digest(np.asarray(guess,dtype="<f8").tobytes()),
            "initial_guess_kind":"zero" if guess is None else "supplied_pressure",
            "solver_s":.01,"verification_s":.002,"quality":q}


def corrected_fixture():
    return {"qualified":True,"outcome":"direct_acceptance","proposal_rejection":None,
            "warm":None,"reset":None,"any_pcg":False,"reset_fallback":False,"elapsed_s":.001}


def case_fixture(zero=False):
    return {"zero_residual_base_retained":zero,"native_base_residual_l2":0.,"native_base_residual_max":0.}


def result_fixture(request):
    temporal=corrected_fixture();temporal["comparison_role"]="bounded32-step temporal diagnostic with one p_base reset; not the full-budget temporal baseline"
    full=solve_fixture(request,3000,request["previous_pressure"])
    full["comparison_role"]="temporal PCG baseline: one fresh full-budget attempt from previous accepted pressure"
    return {"schema":"rheon-global-pressure-teacher-diagnostic-v1",
            "scope":"teacher-state development snapshots; no autonomous rollout or acceleration qualification",
            "cost_scope":"native diagnostic includes solver/correction/verification; excludes model inference and generic custody",
            "state_advanced":False,"native_elapsed_s":.1,"reference":quality_fixture(),
            "raw_learned":quality_fixture(),"raw_temporal":quality_fixture(),"raw_zero":quality_fixture(),
            "learned_jacobi":corrected_fixture(),"temporal_jacobi":temporal,
            "temporal_full_budget":full,"zero_jacobi":solve_fixture(request,3000,None)}


class AdmissionTests(unittest.TestCase):
    def test_strict_json_rejects_duplicate_nonfinite_and_overflow(self):
        for blob in (b'{"a":1,"a":2}',b'{"x":NaN}',b'{"x":1e999}'):
            with self.subTest(blob=blob),self.assertRaises(a.AdmissionError):a.strict_json(blob)

    def test_closed_typed_receipt_and_exact_gn_variant(self):
        value=receipt_fixture();a.validate_receipt(value,"restore")
        mutations=[("optimizer_updates",False),("trained_optimizer_updates",200.),
                   ("parameter_count",372377.),("schema","rheon-global-pressure-training-receipt-v1"),
                   ("extra",1)]
        for key,new in mutations:
            bad=copy.deepcopy(value);bad[key]=new
            with self.subTest(key=key),self.assertRaises(a.AdmissionError):a.validate_receipt(bad,"restore")
        for key,new in (("groups",4.),("affine",0),("epsilon",1e-4)):
            bad=copy.deepcopy(value);bad["architecture"]["block_normalization"][key]=new
            with self.subTest(key=key),self.assertRaises(a.AdmissionError):a.validate_receipt(bad,"restore")

    def test_shape_bool_and_numeric_witness_int32_rejected(self):
        value=receipt_fixture()
        for key,new in (("shape",[True,16,16,16]),("dtype","<f4")):
            bad=copy.deepcopy(value);bad["predictions_inventory"]["pressure"][key]=new
            with self.subTest(key=key),self.assertRaises(a.AdmissionError):a.validate_receipt(bad,"restore")
        value["predictions_inventory"]["row_indices"]["dtype"]="<i4"
        with self.assertRaises(a.AdmissionError):a.validate_receipt(value,"restore")

    def test_fixed_population_types_and_order(self):
        value=receipt_fixture();value["witness_cases"][0]["frame"]=False
        with self.assertRaises(a.AdmissionError):a.validate_receipt(value,"restore")
        cases=a.population();self.assertEqual([c["row_index"] for c in cases],
             [t*24+f for t in range(16) for f in (0,23)])
        self.assertEqual(cases[-1]["case_id"],"smoke-validation-100103:frame-023")

    def test_numeric_archive_roundtrip_without_pickle(self):
        blob,inv=archive_fixture();self.assertEqual(a.decode_npz(blob,inv)["field"].tolist(),[1.,-2.])

    def test_npz_header_shape_and_dtype_changes_with_identical_body_fail(self):
        blob,inv=archive_fixture()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:body=z.read("field.npy")
        for old,new in ((b"'<f8'",b"'<i8'"),(b"(2,)",b"(1,)")):
            changed=body.replace(old,new);out=io.BytesIO()
            with zipfile.ZipFile(out,"w") as z:z.writestr("field.npy",changed)
            with self.subTest(new=new),self.assertRaises(a.AdmissionError):a.decode_npz(out.getvalue(),inv)

    def test_npz_duplicate_extra_compressed_and_nonfinite_rejected(self):
        blob,inv=archive_fixture()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:body=z.read("field.npy")
        for names,compression in ((["field.npy","field.npy"],zipfile.ZIP_STORED),
                                   (["field.npy","extra.npy"],zipfile.ZIP_STORED),
                                   (["field.npy"],zipfile.ZIP_DEFLATED)):
            out=io.BytesIO()
            with zipfile.ZipFile(out,"w",compression=compression) as z:
                for name in names:z.writestr(name,body)
            with self.assertRaises(a.AdmissionError):a.decode_npz(out.getvalue(),inv)
        blob,inv=archive_fixture(np.array([np.nan,0.],dtype="<f8"))
        with self.assertRaises(a.AdmissionError):a.decode_npz(blob,inv)

    def test_bounded_header_precedes_array_allocation(self):
        blob,inv=archive_fixture()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:body=z.read("field.npy")
        body=body[:8]+b'\xff\xff'+body[10:];out=io.BytesIO()
        with zipfile.ZipFile(out,"w") as z:z.writestr("field.npy",body)
        with patch.object(a.np,"frombuffer",side_effect=AssertionError("allocation before header")):
            with self.assertRaises(a.AdmissionError):a.decode_npz(out.getvalue(),inv)

    def test_zero_input_base_bytes_including_signed_zero(self):
        base=np.zeros(4096,dtype="<f8");base[1]=-0.
        residual=np.zeros_like(base);self.assertTrue(a.zero_disposition(base,base,residual))
        bad=base.copy();bad[1]=0.
        with self.assertRaises(a.AdmissionError):a.zero_disposition(bad,base,residual)
        bad=base.copy();bad[3]=1.
        with self.assertRaises(a.AdmissionError):a.zero_disposition(bad,base,residual)

    def test_nonzero_residual_preserves_candidate_and_underflow_rejects(self):
        base=np.zeros(4096,dtype="<f8");candidate=base.copy();candidate[1]=3.
        residual=base.copy();residual[2]=1.
        before=candidate.tobytes();self.assertFalse(a.zero_disposition(candidate,base,residual))
        self.assertEqual(candidate.tobytes(),before)
        residual[2]=1e-200
        with self.assertRaises(a.AdmissionError):a.zero_disposition(candidate,base,residual)

    def test_read_budget_counts_repeated_reads_and_postverification(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"input";p.write_bytes(b"12345")
            inputs=a.Inputs(time.monotonic()+10);inputs.read(p,5);inputs.read(p,5)
            self.assertEqual(inputs.bytes_read,10)
            with patch.object(a,"INPUT_LIMIT",14):
                with self.assertRaises(a.AdmissionError):inputs.read(p,5)

    def test_read_refuses_symlink_and_drift(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"input";p.write_bytes(b"12345");link=Path(root)/"link";link.symlink_to(p)
            inputs=a.Inputs(time.monotonic()+10)
            with self.assertRaises(a.AdmissionError):inputs.read(link,5)
            inputs.read(p,5);p.write_bytes(b"54321")
            with self.assertRaises(a.AdmissionError):inputs.read(p,5)

    def test_expired_deadline_before_read(self):
        with self.assertRaises(a.AdmissionError):a.Inputs(time.monotonic()-1).read(HERE/"admission.py")

    def test_postnative_exact_contract_and_zero_base_proof(self):
        request=request_fixture();request["candidate_pressure"]=request["previous_pressure"][:]
        result=result_fixture(request);case=case_fixture(True)
        a.validate_result(request,result,case)
        result["raw_learned"]["gauge_cell"]=1.
        with self.assertRaises(a.AdmissionError):a.validate_result(request,result,case)

    def test_postnative_original_b_scale_and_state_types(self):
        request=request_fixture();result=result_fixture(request)
        request["original_rhs"][1]=1.
        with self.assertRaises(a.AdmissionError):a.validate_result(request,result,case_fixture())
        request=request_fixture();result=result_fixture(request);result["state_advanced"]=0
        with self.assertRaises(a.AdmissionError):a.validate_result(request,result,case_fixture())

    def test_fresh_nonzero_base_reset_not_zero_or_failed_guess(self):
        request=request_fixture();raw=quality_fixture();raw["qualified"]=False
        raw["residual_l2"]=1.;raw["residual_max"]=1.
        raw["predicted_divergence_max_s_inv"]=request["dt_s"]/np.prod(request["spacing_m"])
        raw["failures"]=["original_rhs_full_residual","predicted_divergence"]
        warm=solve_fixture(request,32,request["candidate_pressure"],False)
        reset=solve_fixture(request,3000,request["previous_pressure"])
        policy={"qualified":True,"outcome":"temporal_reset_fallback","proposal_rejection":None,
                "warm":warm,"reset":reset,"any_pcg":True,"reset_fallback":True,"elapsed_s":.1}
        a.validate_corrected(policy,request,raw,raw,request["candidate_pressure"])
        for guess in (None,request["candidate_pressure"]):
            bad=copy.deepcopy(policy);bad["reset"]=solve_fixture(request,3000,guess)
            with self.assertRaises(a.AdmissionError):a.validate_corrected(bad,request,raw,raw,request["candidate_pressure"])

    def test_failed_quality_cannot_be_labeled_success(self):
        request=request_fixture();s=solve_fixture(request,3000,request["previous_pressure"])
        s["quality"]["courant"]=.6
        with self.assertRaises(a.AdmissionError):a.validate_solve(s,request,3000,request["previous_pressure"])
        s=solve_fixture(request,32,request["candidate_pressure"]);s["iterations"]=33
        with self.assertRaises(a.AdmissionError):a.validate_solve(s,request,32,request["candidate_pressure"])

    def test_zero_start_hash_is_native_full_f64_zero_vector_not_null(self):
        request=request_fixture();s=solve_fixture(request,3000,None)
        self.assertEqual(s["initial_guess_sha256"],"c35020473aed1b4642cd726cad727b63fff2824ad68cedd7ffb73c7cbd890479")
        a.validate_solve(s,request,3000,None)
        s["initial_guess_sha256"]=None
        with self.assertRaises(a.AdmissionError):a.validate_solve(s,request,3000,None)

    def test_missing_or_stale_attestation_authority_rejected(self):
        reader=a.import_pinned(a.LAB/"experiments/global-pressure-v1/training/data.py",a.READER_SHA,
                               a.Inputs(time.monotonic()+10),"gn_test_packet_reader")
        with self.assertRaises(ValueError):reader.validate_residual_origin({},a.RESIDUAL_ANCHOR,{})
        with self.assertRaises(ValueError):reader.validate_residual_origin({},None,{})

    def test_all_reported_physics_gates_and_failure_flags(self):
        request=request_fixture()
        mutations=(("residual_l2",1.),("predicted_divergence_max_s_inv",1.),
                   ("actual_divergence_max_s_inv",1.),("wall_normal_flux_exact_zero",False),
                   ("courant",.6),("gauge_cell",1.),("relative_residual",0.),("rhs_l2",1.))
        for field,value in mutations:
            q=quality_fixture();q[field]=value
            with self.subTest(field=field),self.assertRaises(a.AdmissionError):a.validate_quality(q,request)
        for reason in ("original_rhs_full_residual","predicted_divergence","actual_f32_divergence",
                       "wall_normal_flux","courant","net_volume_flux"):
            q=quality_fixture();q["qualified"]=False;q["failures"]=[reason]
            with self.subTest(reason=reason),self.assertRaises(a.AdmissionError):a.validate_quality(q,request)

    def test_solver_error_does_not_accept_qualified_scratch_and_terminal_is_typed(self):
        request=request_fixture();solve=solve_fixture(request,3000,request["previous_pressure"])
        solve["solver_error"]="breakdown"
        with self.assertRaises(a.AdmissionError):a.validate_solve(solve,request,3000,request["previous_pressure"])
        reset=solve_fixture(request,3000,request["previous_pressure"],False)
        raw={"qualified":False,"failures":["prediction_gauge"]}
        policy={"qualified":False,"outcome":"terminal_failure","proposal_rejection":"prediction_gauge",
                "warm":None,"reset":reset,"any_pcg":True,"reset_fallback":True,"elapsed_s":.01}
        a.validate_corrected(policy,request,raw,quality_fixture(),request["candidate_pressure"])
        policy["qualified"]=True
        with self.assertRaises(a.AdmissionError):a.validate_corrected(policy,request,raw,quality_fixture(),request["candidate_pressure"])

    def test_nonzero_attested_norm_and_max_mismatch_rejected_for_every_case(self):
        request=request_fixture();result=result_fixture(request)
        for metric in ("native_base_residual_l2","native_base_residual_max"):
            case=case_fixture();case[metric]=1.
            with self.subTest(metric=metric),self.assertRaises(a.AdmissionError):a.validate_result(request,result,case)
        residual=np.zeros(4096,dtype="<f8");residual[:2]=[3.,-4.]
        self.assertEqual(a.residual_metrics(residual),(5.,4.))

    def test_all17_typed_historical_artifact_bindings(self):
        root=a.LAB/"runs/global-pressure-groupnorm-v1"
        inputs=a.strict_json((root/"restore-stage/fresh-preflight.json").read_bytes())["material"]["inputs"]
        report=a.strict_json((a.LAB/"docs/plans/global-pressure-groupnorm-v1/reports/training.json").read_bytes())
        origin=receipt_fixture()["originating_training_run"]
        a.validate_originating_inputs(inputs,report,origin)
        for field,value in (("artifact_bytes",True),("schema_version",True),("run_id","wrong"),
                            ("measured_sha256","0"*64)):
            bad=copy.deepcopy(inputs);bad["trained_weights"]["source"][field]=value
            with self.subTest(field=field),self.assertRaises(a.AdmissionError):a.validate_originating_inputs(bad,report,origin)
        bad=copy.deepcopy(inputs);bad.pop("trained_checkpoint_000")
        with self.assertRaises(a.AdmissionError):a.validate_originating_inputs(bad,report,origin)

    def test_postverification_accounts_reads_even_on_hash_failure(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);p=root/"input";p.write_bytes(b"first")
            inputs=a.Inputs(time.monotonic()+10);inputs.read(p,5)
            admitted={"_inputs":inputs,"consumed_files":inputs.files,"proof":{},"_prepared":root,
                      "_prepared_modes":{"input":"100644"}}
            a.verify_unchanged(admitted,time.monotonic()+10)
            self.assertEqual(admitted["proof"]["input_bytes_read"],10)
            p.write_bytes(b"other")
            with self.assertRaises(a.AdmissionError):a.verify_unchanged(admitted,time.monotonic()+10)
            self.assertEqual(admitted["proof"]["input_bytes_read"],15)

    def test_prepared_new_file_mode_and_symlink_drift_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);p=root/"file";p.write_bytes(b"x")
            expected={"file":"100644"};a.prepared_modes(root,expected,time.monotonic()+10)
            p.chmod(0o755)
            with self.assertRaises(a.AdmissionError):a.prepared_modes(root,expected,time.monotonic()+10)
            p.chmod(0o644);(root/"extra").write_bytes(b"y")
            with self.assertRaises(a.AdmissionError):a.prepared_modes(root,expected,time.monotonic()+10)
            (root/"extra").unlink();(root/"link").symlink_to(p)
            with self.assertRaises(a.AdmissionError):a.prepared_modes(root,expected,time.monotonic()+10)


if __name__=="__main__":unittest.main()
