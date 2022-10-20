# lol

- Simpletrains (GPU5)
  - SNN - !!!
  - 3DCNN - OK
  - CNN - OK
- SimpleTrains with new transforms
  - EventDrop2, EventDrop3, Transrot
- New experiments of SSL: (GPU6 & Douai)
  - New transforms mixed with ...
    - EventDrop2 (+cutpaste)
      - SNN
      - 3DCNN
      - CNN
      - SCNN
      - 3DSCNN
    - TransRot (Dyn/Stat transform TODO)
      - SNN
      - 3DCNN
      - CNN
      - SCNN
      - 3DSCNN
    - EventDrop3 (+ moving occ) (wait for eventdrop2)
      - SNN
      - 3DCNN
      - CNN
      - SCNN
      - 3DSCNN
  - Finetune params of main_ssl for the best evals

- T-SNE of finetuned & simple_trains

- Best SSL with DVS-Lip
- DVS-Lip + Finetune