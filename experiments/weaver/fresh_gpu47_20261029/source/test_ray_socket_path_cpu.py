"""Use installed Ray's actual Unix-socket validator on the launcher paths."""
from pathlib import Path
import unittest
from ray._private.utils import validate_socket_filepath

class RaySocketBudget(unittest.TestCase):
 def test_observed_failed_long_socket_is_rejected(self):
  path='/dev/shm/weaverintegrated_1004_2169536_28166/session_2026-10-27_11-28-57_148484_2169602/sockets/plasma_store'
  self.assertEqual(len(path.encode()),108)
  with self.assertRaisesRegex(OSError,'107 bytes'):validate_socket_filepath(path)

 def test_actual_launcher_short_default_fits_worst_case_identity_fields(self):
  script=(Path(__file__).parent/'run.sh').read_text()
  line=next(x for x in script.splitlines() if x.startswith('export ACCERL_RAY_TMP='))
  template=line.split(':-',1)[1].rsplit('}',1)[0]
  for uid,pid,random in [('1004','2169536','28166'),('4294967295','2147483647','32767')]:
   base=template.replace('${UID}',uid).replace('$$',pid).replace('${RANDOM}',random)
   path=base+'/session_2026-10-27_11-28-57_148484_'+pid+'/sockets/plasma_store'
   self.assertLessEqual(len(path.encode()),107)
   validate_socket_filepath(path)

if __name__=='__main__':unittest.main()
