import queue
import threading
import json
from pathlib import Path
import numpy as np


class RepresentationLearner:
    def __init__(self, module, capacity=50000, batch_size=256, min_size=256,
                 seed=0, gamma=.99, phi_weight=1., dyn_weight=1.):
        self.module = module
        self.capacity, self.batch_size = max(2,int(capacity)), max(2,int(batch_size))
        self.min_size = min(self.capacity, max(2,int(min_size)))
        self.rng = np.random.default_rng(seed)
        self.options = dict(gamma=gamma,phi_weight=phi_weight,dyn_weight=dyn_weight)
        self.queue = queue.Queue(maxsize=8)
        self.lock = threading.RLock()
        self.error = None
        self.closed = False
        self.size = self.position = 0
        self.arrays = None
        self.worker_reward_moments = {}
        self.latest = dict(phi_loss=0.,dyn_loss=0.,prediction_error=0.,mean_d_phi=0.,
            mean_metric_target=0.,analytical_distance=0.,task_reward_difference=0.,
            successor_distance=0.,reward_prediction_loss=0.,representation_updates=0)
        self.thread = threading.Thread(target=self._run, name='mira-representation-learner', daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError('MIRA representation learner failed') from self.error

    def submit(self, states, actions, task_rewards, next_states, valid_action_masks,
               state_keys=None, next_terminals=None, next_state_keys=None):
        self.check()
        if self.closed:
            raise RuntimeError('MIRA learner is closed')
        states, actions, following = [np.asarray(x).copy() for x in (states, actions, next_states)]
        n = len(states)
        masks = np.asarray(valid_action_masks, dtype=bool)
        if masks.shape != (n, self.module.reference_action_dim) or not masks.any(axis=1).all():
            raise ValueError('Executed transitions require nonempty legal-action masks')
        rewards = np.asarray(task_rewards, dtype=np.float32).reshape(n, 1)
        if state_keys is None:
            from .reference_policy import state_identity
            state_keys = [state_identity(s.tolist(), m.tolist()) for s, m in zip(states, masks)]
        keys = np.asarray(state_keys, dtype='S64').reshape(n)
        terminal = np.zeros(n, bool)
        done = np.zeros(n, bool) if next_terminals is None else np.asarray(next_terminals, bool).reshape(n)
        batch = [states, actions, rewards, following, masks, terminal, keys]
        if done.any():
            if next_state_keys is None:
                from .reference_policy import state_identity
                next_state_keys = [state_identity(s.tolist(), 'absorbing') for s in following]
            terminal_rows = [following[done], np.zeros_like(actions[done]), np.zeros_like(rewards[done]),
                             following[done], np.zeros_like(masks[done]), np.ones(done.sum(), bool),
                             np.asarray(next_state_keys, dtype='S64').reshape(n)[done]]
            batch = [np.concatenate([left, right]) for left, right in zip(batch, terminal_rows)]
        batch = tuple(batch)
        while True:
            self.check()
            try:
                self.queue.put(batch,timeout=.2)
                return self.stats()
            except queue.Full:
                continue

    def _store(self, batch):
        if len({len(x) for x in batch}) != 1 or not all(np.isfinite(x).all() for x in batch[:-1]):
            raise ValueError('Invalid representation replay batch')
        if self.arrays is None:
            self.arrays = [np.empty((self.capacity,)+x.shape[1:],dtype=x.dtype) for x in batch]
        count = min(len(batch[0]),self.capacity)
        ids = (np.arange(count)+self.position)%self.capacity
        for buffer, values in zip(self.arrays,batch):
            buffer[ids] = values[-count:]
        self.position = (self.position+count)%self.capacity
        self.size = min(self.capacity,self.size+count)

    def _run(self):
        while True:
            batch = self.queue.get()
            try:
                if batch is None:
                    return
                if self.error is not None:
                    continue
                with self.lock:
                    self._store(batch)
                    if self.size >= self.min_size:
                        ids = self.rng.choice(self.size,min(self.size,self.batch_size),replace=False)
                        states, actions, rewards, following, masks, terminals, keys = [a[ids] for a in self.arrays]
                        if hasattr(self.module,'encode_actions'):
                            actions = self.module.encode_actions(actions)
                        self.latest = self.module.train_batch(states,actions,rewards,following,
                            pair_indices=self.rng.permutation(len(ids)), valid_action_masks=masks,
                            terminals=terminals, state_keys=keys, reference_rng=self.rng, **self.options)
            except BaseException as error:
                self.error = error
            finally:
                self.queue.task_done()

    def stats(self):
        self.check()
        with self.lock:
            return dict(self.latest,replay_size=self.size)

    def weights(self):
        self.check()
        with self.lock:
            return self.module.get_weights()

    def embed(self, states):
        self.check()
        with self.lock:
            return self.module.embed(states)

    def record_reward_state(self, worker_id, state):
        with self.lock:
            self.worker_reward_moments[str(worker_id)] = dict(state)

    def save_report(self, directory):
        self.check()
        with self.lock:
            report = dict(self.latest, replay_size=self.size,
                core_version=self.module.get_weights()['core_version'],
                statistic_indices=self.module.statistic_indices.tolist(),
                lambda_ana=self.module.lambda_ana, mc_samples=self.module.mc_samples,
                worker_reward_moments=dict(self.worker_reward_moments))
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'representation_training.json').write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    def close(self):
        if not self.closed:
            self.closed = True
            self.queue.put(None)
            self.thread.join()
        self.check()
