import unittest

from pyvisc.schemes import SchemeVarH

class TestRenderVarH(unittest.TestCase):
    def test_render_varh(self):
        dct = {
            "precision": "single",
            "NX": 10,
            "NY": 20,
            "NZ": 30,
            "tau": 0.867,
            "vel_set": "D3Q19",
            "use_ibm": True,
            "non_newtonian_model": "BINGHAM",
            "sim_id": 1,
            "path_save": "teste",
            "steps": 2000,
            "macr_save": 200,
            "data_report": 300,
            "FX": 1e-4,
            "FY": 1e-5,
            "FZ": 1e-3,
            "data_stop": True,
            "data_save": True,
            "pop_save":  True,
            "ini_step":  10,
            "n_gpus": 2,
            "resid_max": 1e-5,
        }
        SchemeVarH(**dct).render(path_save="tests/generated")



if __name__ == "__main__":
    unittest.main()