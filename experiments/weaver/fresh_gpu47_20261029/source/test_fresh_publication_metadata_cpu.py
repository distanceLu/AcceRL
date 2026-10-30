"""Actual runner collection from SQLite TEXT, with strict publication validation.

Only unrelated rank/placement queries and the runtime service-plan import are
stubbed. The database schema affinity, event selection, JSON materialization,
runner collect function and final publication validator are real CPU paths.
"""
import ast
import copy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from world_backend_audit import validate_online_publication


SOURCE = Path(__file__).resolve().parent


def runner_functions():
    tree = ast.parse((SOURCE / 'runner.py').read_text(encoding='utf8'))
    names = {'_publication_version_from_sqlite', 'collect', 'measurement_configuration'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise AssertionError('The actual runner boundary functions must remain present')
    namespace = {'json':json, 'sqlite3':sqlite3, 'time':time, 'SOURCE':SOURCE,
        'stamp':lambda:'cpu-fixture', 'safe':SimpleNamespace(
            rank_updates=lambda *args:{'policy':{'global_updates':1240},'ctrl':{'global_updates':248}},
            placement_evidence=lambda *args,**kwargs:{})}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual runner database boundary','exec'),namespace)
    return namespace


def details(version):
    inventory = {'kind':'ctrl','version':version,'hash':'a'*64,
        'publication':{'parameter_names':['upstream.wm.weight','upstream.rm.weight'],
            'bundle_metadata':{'reward_provider':'world_head'}}}
    return {'kind':'ctrl','transport':'cuda_ipc','receiver_ranks':[0,1,2,3],
        'inventories':[copy.deepcopy(inventory) for _ in range(4)],'ack_completed_ns':60,
        'driver_timing':{'rank_ACK_observed_ns':[20,21,22,23],'release_ready_ns':50}}


class PublicationSQLiteBoundary(unittest.TestCase):
    def collect(self, value=248, *, inventory_version=248):
        functions=runner_functions()
        plan=SimpleNamespace(active_kinds=('policy','ctrl'),backend='weaver',world_trainable=True,
            to_dict=lambda:{'backend':'weaver'})
        runtime=ModuleType('world_backends.runtime')
        runtime.service_plan_from_state=lambda state:plan
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'training').mkdir()
            db=root/'training/collection.sqlite'
            with closing(sqlite3.connect(db)) as connection:
                connection.execute('CREATE TABLE events (id INTEGER PRIMARY KEY,run_id TEXT,ts_ns INTEGER,'
                    'pid INTEGER,actor TEXT,actor_id TEXT,model_version TEXT,details_json TEXT,workload TEXT,kind TEXT)')
                connection.execute('CREATE TABLE replay_samples (run_id TEXT,replay TEXT,resident INTEGER,'
                    'policy_uses INTEGER,ctrl_uses INTEGER,reward_uses INTEGER)')
                connection.execute('INSERT INTO events VALUES (1,?,?,?,?,?,?,?,?,?)',
                    ('run',100,10,'driver','0',value,json.dumps(details(inventory_version)),None,
                     'weight_publication_committed'))
                stored=connection.execute('SELECT model_version,typeof(model_version) FROM events').fetchone()
                connection.commit()
            state={'run_id':'run','state':'stopped_by_user','mode':'manual',
                'world_backend_config':{'backend_options':{'transport':'isolated_online_ddp'}}}
            with patch.dict(sys.modules,{'world_backends.runtime':runtime}):
                report=functions['collect'](root,state)
            return report,stored

    def test_actual_integer_writer_becomes_sqlite_text_then_strict_valid_receipt(self):
        report,stored=self.collect(248)
        self.assertEqual(stored,('248','text'))
        row=report['world_publication_committed'][0]
        self.assertIs(type(row['model_version']),int)
        self.assertEqual(row['model_version'],248)
        self.assertEqual(row['details'],details(248))
        validated=validate_online_publication(report['world_publication_committed'],version=248)
        self.assertTrue(validated['all_four_receiver_ACKs'])
        self.assertTrue(validated['all_four_producer_releases'])

    def test_zero_version_survives_sqlite_and_json_roundtrip(self):
        report,stored=self.collect(0,inventory_version=0)
        self.assertEqual(stored,('0','text'))
        rows=json.loads(json.dumps(report))['world_publication_committed']
        self.assertIs(type(rows[0]['model_version']),int)
        self.assertEqual(validate_online_publication(rows,version=0)['version'],0)

    def test_corrupt_or_noncanonical_sqlite_text_rejected(self):
        for value in ('-1','01','+1','1.0','1e2',' 1','1 ','\n1','','１２',None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.collect(value)

    def test_boundary_rejects_boolean_float_and_integer_subclasses(self):
        class IntegerSubclass(int):pass
        decode=runner_functions()['_publication_version_from_sqlite']
        for value in (True,False,248.,float('nan'),float('inf'),-1,IntegerSubclass(248),b'248'):
            with self.subTest(value=repr(value)),self.assertRaises(ValueError):decode(value)
        for value in (0,248,'0','248'):
            self.assertIs(type(decode(value)),int)

    def test_only_sqlite_column_decoded_inventory_validation_stays_strict(self):
        report,_=self.collect(248,inventory_version='248')
        self.assertIs(type(report['world_publication_committed'][0]['model_version']),int)
        with self.assertRaises(RuntimeError):
            validate_online_publication(report['world_publication_committed'],version=248)
        report,_=self.collect(248)
        report['world_publication_committed'][0]['model_version']='248'
        with self.assertRaises(RuntimeError):
            validate_online_publication(report['world_publication_committed'],version=248)


if __name__=='__main__':unittest.main()
