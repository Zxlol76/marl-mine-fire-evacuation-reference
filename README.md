# Mine-fire evacuation MARL reference implementation

This repository provides reference implementations of the learning and physical-model components described in the paper:

- an edge-aware message passing neural network (MPNN);
- a masked, parameter-shared actor with a two-layer policy head;
- separate critic graph encoders and agent-ID embeddings;
- independent and centralized critic variants illustrating IPPO and MAPPO;
- clipped PPO, per-agent advantage normalization, valid-agent masking, and
  generalized advantage estimation (GAE);
- multi-agent rollout storage and IPPO/MAPPO update structure;
- the fitted smoke-spread velocity model and directional graph propagation;
- gradient-, visibility-, stamina-, and equipment-aware worker movement;
- crew-level refuge chamber reservation and occupancy management;
- initial decision locking and event-triggered replanning control;
- paper hyperparameter configuration.
