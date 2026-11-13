# EGL environment rendering

User authorization: “接入EGL”, following the agreed sequence of EGL integration first, workload admission experiments later.

The managed recipe will use NVIDIA EGL for the existing Real and online Eval environments. Physics stays on CPU. Real remains limited to 150 actions; Eval remains at 220. Worker counts, sampling, rewards, updates and evaluation budgets are unchanged.

Render workers keep CUDA hidden and select EGL independently. Match EGL devices through DRM/PCI identity to the allocated NVIDIA GPU UUID; never assume CUDA, physical GPU and EGL indices coincide. Follow the Policy inference route: Real uses replica 0 and the three Eval workers use replicas 1–3. Model actors must not inherit an EGL device selector.

Configure the backend before simulator/OpenGL imports. After environment construction, verify the active display's PCI identity and NVIDIA renderer, and persist a per-worker receipt. Refuse incorrect placement or software fallback. Keep an explicit OSMesa configuration for comparison/rollback. Shader caches and validation artifacts belong under the experiment data directory.

Approaches considered: changing only MUJOCO_GL would leave multi-GPU placement unverified; replacing the physics backend exceeds scope. Explicit per-worker EGL placement reuses the existing simulator and measured working backend.

Validate pure placement/configuration contracts locally, then run bounded fixed-action environments through Ray on allocated GPUs 4–7. Do not modify installed libraries. A code deployment cannot change already-running process contexts. The live training is pinned to fb59cd2 and has no exact optimizer/Replay resume; stopping it is a separate lifecycle decision.
