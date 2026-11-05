import unittest
from test_fresh_wrapper_cpu import admitted_config
import training_entry as t

class TrajectoryAdmissionTests(unittest.TestCase):
    def test_only_complete_trajectory_profile_admitted(self):
        b,p,a=admitted_config()
        self.assertEqual(t.BACKEND_CONFIG,'examples/trajectory_01_imagined.json')
        self.assertTrue(t.validate_backend_admission(b,p,a))
        for value in ('chunks',None,True):
            b,p,a=admitted_config();b['backend_options']['fresh']['policy_sampling']=value
            with self.assertRaises(ValueError):t.validate_backend_admission(b,p,a)
    def test_missing_sampling_flag_fails_instead_of_silently_using_parent(self):
        b,p,a=admitted_config();del b['backend_options']['fresh']['policy_sampling']
        with self.assertRaises(ValueError):t.validate_backend_admission(b,p,a)

if __name__=='__main__':unittest.main()
