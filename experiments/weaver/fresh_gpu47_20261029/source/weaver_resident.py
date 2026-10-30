"""Bounded, child-owned rollout state. No Torch import or tensor serialization."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import hmac
import math
import secrets


class ResidentStateError(ValueError):
    """A lost/expired state requires a fresh real context, never a fallback."""
    def __init__(self, reason):
        super().__init__('Stale WEAVER rollout/cache version; resident ' + reason)


def retained_bytes(value):
    """Count unique backing storage, including views that retain a larger array."""
    visited, storage = set(), set()
    def count(item, depth=0):
        if depth > 64:
            raise ResidentStateError('state nesting exceeds the admitted bound')
        if item is None or isinstance(item, (bool, int, float)):
            return 8
        if isinstance(item, str):
            return len(item.encode('utf8'))
        if isinstance(item, bytes):
            return len(item)
        identity = id(item)
        if identity in visited:
            return 0
        visited.add(identity)
        # Tensor storage metadata does not transfer data or initialize CUDA.
        if hasattr(item, 'untyped_storage'):
            backing = item.untyped_storage()
            key = (str(item.device), int(backing.data_ptr()), int(backing.nbytes()))
            if key in storage:
                return 0
            storage.add(key)
            return key[-1]
        if type(item).__module__.split('.')[0] == 'numpy':
            import numpy as np
            if isinstance(item, np.ndarray):
                if item.dtype.hasobject:
                    raise ResidentStateError('object arrays are not admitted')
                backing = item
                while isinstance(backing.base, np.ndarray):
                    backing = backing.base
                key = ('numpy', int(backing.__array_interface__['data'][0]), int(backing.nbytes))
                if key in storage:
                    return 0
                storage.add(key)
                return key[-1]
            if isinstance(item, np.generic):
                return int(item.dtype.itemsize)
        if isinstance(item, dict):
            return sum(count(k, depth+1)+count(v, depth+1) for k,v in item.items())
        if isinstance(item, (list, tuple)):
            return sum(count(v, depth+1) for v in item)
        raise ResidentStateError('unsupported retained state type ' + type(item).__name__)
    return count(value)


class ResidentRolloutOperations:
    """Wrap GPU operations; only opaque capabilities and public outputs leave."""
    protocol = 'weaver_resident_v1'
    _identity_keys = {'protocol','run_id','owner','actor_id','version','session','generation'}
    _initial_keys = ('initial_reward_score','initial_success_probability','initial_terminal',
                     'reward_semantics','reward_calibrated')

    def __init__(self, operations, run_id, actor_id, owner_identity, *, max_entries=8,
                 max_bytes=64*1024*1024, max_requests=256, max_actions=4096):
        if not run_id or not isinstance(run_id,str) or len(run_id)>256:
            raise ValueError('Resident owner requires an exact bounded run ID')
        if type(actor_id) is not int or actor_id < 0:
            raise ValueError('Resident actor ID must be a nonnegative integer')
        self.owner = {k:owner_identity[k] for k in ('pid','uid','birth')}
        if any(type(v) is not int or v < 0 for v in self.owner.values()):
            raise ValueError('Resident owner requires exact PID/UID/birth integers')
        for value,maximum in [(max_entries,128),(max_bytes,1024**3),
                              (max_requests,4096),(max_actions,65536)]:
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError('Resident limits must be positive bounded integers')
        if not callable(getattr(operations,'init_resident_rollout',None)) or not callable(getattr(operations,'request_wire',None)):
            raise ValueError('Resident inference requires explicit owned-state operations')
        self.operations, self.run_id, self.actor_id = operations,run_id,actor_id
        self.max_entries,self.max_bytes = max_entries,max_bytes
        self.max_requests,self.max_actions = max_requests,max_actions
        self.max_chunk_actions=getattr(getattr(operations,'backend',None),'horizon',8)
        if type(self.max_chunk_actions) is not int or not 1<=self.max_chunk_actions<=256:
            raise ValueError('Resident chunk must have an admitted bounded backend horizon')
        self.version = operations.published_version
        self.entries = OrderedDict()
        self._secret=secrets.token_bytes(32)
        self.bytes = 0
        self.evictions = 0

    def _handle(self, session, entry):
        return {'backend':'weaver','version':self.version,'_resident':{
            'protocol':self.protocol,'run_id':self.run_id,'owner':dict(self.owner),
            'actor_id':self.actor_id,'version':self.version,'session':session+self._mac(session,entry['generation']),
            'generation':entry['generation']}}

    def _mac(self, session, generation):
        value=f'{session}:{generation}:{self.version}:{self.run_id}:{self.actor_id}'.encode()
        return hmac.new(self._secret,value,hashlib.sha256).hexdigest()[:32]

    def _validate(self, handle, *, require_present=True):
        if (not isinstance(handle,dict) or handle.get('backend')!='weaver' or type(handle.get('version')) is not int
                or set(handle)-({'backend','version','_resident'}|set(self._initial_keys))):
            raise ResidentStateError('handle is not an admitted capability')
        token = handle.get('_resident')
        if not isinstance(token,dict) or set(token)!=self._identity_keys:
            raise ResidentStateError('capability fields differ')
        expected = {'protocol':self.protocol,'run_id':self.run_id,'owner':self.owner,
                    'actor_id':self.actor_id,'version':self.version}
        if (any(token.get(k)!=v for k,v in expected.items()) or handle['version']!=self.version
                or not isinstance(token['owner'],dict) or set(token['owner'])!=set(self.owner)
                or any(type(v) is not int for v in token['owner'].values())
                or type(token['actor_id']) is not int or type(token['version']) is not int
                or type(token['generation']) is not int or token['generation']<0
                or not isinstance(token['session'],str) or len(token['session'])!=64):
            raise ResidentStateError('capability belongs to another owner/run/version')
        session=token['session'][:32]
        if not hmac.compare_digest(token['session'][32:],self._mac(session,token['generation'])):
            raise ResidentStateError('capability authentication failed')
        entry=self.entries.get(session)
        if entry is None:
            if require_present:
                raise ResidentStateError('session was released, evicted or never issued')
            return session,None
        if token['generation']!=entry['generation']:
            raise ResidentStateError('handle generation was already consumed')
        if any(key in handle and (type(handle[key]) is not type(value) or handle[key]!=value)
               for key,value in entry['public'].items()):
            raise ResidentStateError('initial public metadata was changed')
        return session,entry

    def _remove(self, session):
        entry=self.entries.pop(session)
        self.bytes-=entry['bytes']
        return entry

    def _store(self, session, entry):
        size=retained_bytes(entry['state'])
        if size > self.max_bytes:
            raise ResidentStateError('single session exceeds the byte budget')
        while self.entries and (len(self.entries)>=self.max_entries or self.bytes+size>self.max_bytes):
            oldest=next(iter(self.entries))
            self._remove(oldest)
            self.evictions+=1
        entry['bytes']=size
        self.entries[session]=entry
        self.bytes+=size

    def clear(self):
        count=len(self.entries)
        self.entries.clear()
        self.bytes=0
        return count

    def set_published_version(self, version):
        if type(version) is not int or version<self.version:
            raise ResidentStateError('publication cannot move backward')
        self.clear()
        self.operations.set_published_version(version)
        self.version=version

    def init_rollout(self, context, published_version=None):
        requested=self.version if published_version is None else published_version
        if type(requested) is not int or requested!=self.version:
            raise ResidentStateError('initialization version differs')
        state=self.operations.init_resident_rollout(context,requested)
        entry={'state':state,'generation':0,'requests':0,'actions':0,
               'public':{key:state[key] for key in self._initial_keys if key in state}}
        session=secrets.token_hex(16)
        self._store(session,entry)
        handle=self._handle(session,entry)
        for key in self._initial_keys:
            if key in state:
                value=state[key]
                if not isinstance(value,(str,bool,int,float)) or isinstance(value,float) and not math.isfinite(value):
                    self._remove(session)
                    raise ResidentStateError('initial public metadata must be finite primitive values')
                handle[key]=value
        return handle

    def request(self, handle, actions, instruction=None):
        session,entry=self._validate(handle)
        if not 1<=len(actions)<=self.max_chunk_actions:
            raise ResidentStateError('action chunk exceeds the admitted backend horizon')
        if entry['requests']>=self.max_requests or entry['actions']+len(actions)>self.max_actions:
            self._remove(session)
            raise ResidentStateError('session lifetime action/request bound reached')
        # Consume before execution: failures cannot retry an ambiguous state or
        # retain an old state after an in-place backend mutation.
        self._remove(session)
        result=self.operations.request_wire(entry['state'],actions,instruction)
        if not isinstance(result,dict) or 'next_state' not in result or result.get('served_version')!=self.version:
            raise ResidentStateError('prediction lacks the current owned next state')
        next_state=result['next_state']
        entry.update(state=next_state,generation=entry['generation']+1,
                     requests=entry['requests']+1,actions=entry['actions']+len(actions))
        self._store(session,entry)
        public={k:v for k,v in result.items() if k!='next_state'}
        public['next_state']=self._handle(session,entry)
        return public

    def release_rollout(self, handle):
        # Strict validation rejects forgery. Already missing legitimate handles
        # are a harmless explicit release (e.g. publication/eviction during RPC).
        session,entry=self._validate(handle,require_present=False)
        if entry is not None:
            self._remove(session)
        return {'released':entry is not None,'resident_entries':len(self.entries),'resident_bytes':self.bytes}

    def report(self):
        return {'protocol':self.protocol,'max_entries':self.max_entries,'max_bytes':self.max_bytes,
                'max_requests':self.max_requests,'max_actions':self.max_actions,
                'max_chunk_actions':self.max_chunk_actions,
                'entries':len(self.entries),'bytes':self.bytes,'evictions':self.evictions,
                'published_version':self.version}
