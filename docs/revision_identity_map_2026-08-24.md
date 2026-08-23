# V3 revision identity map — 2026-08-24

The 39 commits on `experiment-v3-rebuild` were rewritten to use the repository owner's verified
Git identity:

- author and committer name: `o0SilentStorm0o`;
- author and committer email: `davidstrnadel@seznam.cz`.

Only commit identity metadata and the parent hashes derived from it changed. A sequence comparison
confirmed the same number of commits, identical source-tree hashes, and identical commit subjects
before and after the rewrite. The `main` branch was not rewritten.

Raw experiment provenance intentionally retains the old source revision recorded when each run was
created. The table below resolves those immutable provenance values to the identity-corrected
history.

| Original revision | Identity-corrected revision | Commit subject |
| --- | --- | --- |
| `080c948d0d5d76bd0ae46dee93df9e87ce931289` | `cb8385fd71a9aa350ff70865819e4850ee50b751` | Rebuild quantum experiment foundation |
| `c8f59ab367f0e38302bab01bd82f4df89d018f2d` | `f582b15653119adb8c358cf22d942f3b0363b7c8` | Add controlled ablations and image-coupled evaluation |
| `68fe94e86cc2b6569ad5acc9a2886ac4365f71bd` | `29e418c8b928cd7db79d7a2e5b25e6a9ddf1e935` | Add LUMI benchmark and execution plan |
| `1917d0648ab6d9d21a217729533167807b9c8096` | `d760895097fa7cd3189f135fc82e860fdec8e662` | Add frozen contrastive quantum reference experiment |
| `9cf2c61c39766ff576a07880ff3af8fd147bd1d6` | `df6a07184c56cf1484a0f51f044dcb90095bd6d5` | Record contrastive Stage 1 falsification |
| `de259239f2673f429d096194d5ff98d74cfd8f04` | `b1282f6902b1daa2353b63f3f508b947b32b5762` | Freeze post-hoc contrastive failure diagnostics |
| `0c1d08ec916d9b5405623a0f09cbba414227fd61` | `d26bc1047af40ab035c4dbcea0538c2773dba250` | Add contrastive failure diagnostic tooling |
| `1be4a48e538863b6434f78505deb2bc743f032f6` | `65c1be639cca1598ab412f8f20ed91fd7f472ddd` | Verify diagnostic reconstruction against stored metrics |
| `0cf7df5e1bebddcd5430d00ca7b69629a0050dda` | `0674c78110e5c4f82fad7eb589bae97a926e8e9e` | Document causal diagnosis of contrastive failure |
| `b9431fa02b2c6be57456e4a5e4aae27a0a17c8b7` | `a1e7123c51ba518906e801e58e8b46ced3ea72f6` | Freeze trainable coverage development protocol |
| `4bc24e8f09294e6100eea95b8581ce30817f79ce` | `cffdcd3f1e074cf081fa160f2823221653b11a17` | Add trainable relational coverage experiment |
| `22c72c51aa40b5d757e7a5b05074779f8767242d` | `42b5eaeb71186121ef9d085834e3ef0458a84e0d` | Freeze relational coverage gradient calibration |
| `c955516cc8450904b80e214ababb68bf15692b64` | `548784afa9a9331716028dce8a9a0d18852658d1` | Add relational development audit |
| `e19dcbcd37f7db07c3eddf1230da384675ce5113` | `5efd42302ddee20294f59edd51489360d326ef95` | Record relational coverage development gate |
| `9beb9aaf17b68f34e0cffde5cb8f216b2fae11ac` | `8fcddc144d10dac6fd1f21798d4c663b66013053` | Add relational circuit ablation variants |
| `a547d998fef1b9aacdea541785f4709608e0c2dc` | `c04600fd15c17f831f32cb2d5207d5161d311f1f` | Add relational circuit ablation audit |
| `8a181349f9f30904f6b6fdd65da5a7d9e664a555` | `7a8fa9961e55f9de29de323877623e84da59a813` | Record relational circuit ablation failure |
| `d2a490975fefc40ece5e3050f3864329b14a6fd8` | `5afec49480c48cb297d469a071aea5d52cd10fe4` | Freeze gradient-matched circuit diagnostic |
| `26794d637a966dfa42f42cc5c955490bf8d20257` | `194f5d548d3ded9ef82fda13440f0ce61b4661eb` | Extend relational calibration to circuit controls |
| `fbe4028f069c895e44ee6def85b88b1861271e5e` | `2ade56ad9c8b5661ad66a1b0212108aa5ca10c25` | Freeze gradient-matched circuit weights |
| `9760e522d89e73514e07f00a2f3f9c5c183eeb54` | `f82cd4ed67d0c3b12ea4691f72c1844256bb9c08` | Add gradient-matched relational audit |
| `4bd89daf202b15585bbfce74c21915f544cf344d` | `afca60881e1a2f048e8bd291bc3a37a63a681844` | Extend relational mechanism audit |
| `e32400e5df75db0f9bb8ef60f6e840db7bb9c441` | `7ed1460981799007bc41d083329451efd42dd4e2` | Record gradient-matched mechanism diagnosis |
| `d9d8536c044df8b73cc70a05a1798793b74c9287` | `d5751222842fc18726a1c74280ef93016261c7af` | Freeze full-trajectory budget protocol |
| `4a9217410722d2065867eec59709cc8f1b789ee3` | `5e0cd2f57e8b48036e914246a66ed194622b3674` | Control optimizer-level shared budget |
| `be09e84941066ecc09504b818f54d3510949f3ef` | `90225040d70a2a0cdf4962a93dfe43127e967b70` | Add full-trajectory relational budget control |
| `3f6f1e074cb87543e73f314a081e8e22423defda` | `09276b79b1685dc155f99d312f9754f4405b2cdd` | Refine float32 budget application |
| `55a00378f366d317bdee909dbbaa2842bdd6d47c` | `1dcad2699690fa6c427fdbb7b1845c7406e7b408` | Stabilize realized budget correction |
| `847cacfa6d52a24055e529c80d35e7b72a9f5729` | `d6045568ab25eab8e15368d1ec9e98c5f94ed22e` | Match Adam proposal arithmetic |
| `7f5250d05133a44c148190e392e01a1bd965f07f` | `09e5a7ffe02e138f15b3a646c5111eaca446b85f` | Reproduce Adam displacement exactly |
| `fca787fb3f098e1ecea0e4b0b8760458810e1913` | `133e6c9ad2bae477a61bc5f4ff02cbb0b2ecf894` | Bracket realized displacement refinement |
| `a0743f827da86b82d9e6bfdc6bd7a4b62e6f5f68` | `07fb66630a8b824a3164e45629b713d49d4d6acc` | Bound float32 quantization repair |
| `fec0a54cdef257dd4310ba4c1a0b934c240a45f7` | `ef7a7a84c347775e5f1613ff7d0e5c12ca8cc91a` | Stabilize quantized budget realization |
| `dd74c64d4704d2f768865cb5956b61aee755fc54` | `f438ad392b6fb7fc99dd70e5e6caca7393ae0219` | Raise quantization repair guard |
| `245b7003f207c403faebd7df3b123df85f0c1f04` | `35fd8db0ffd95e400583c685cb93fed6fd25f85a` | Allow bounded quantization convergence |
| `e9f4fd1fd25e776d1b64a68998e6bf1577631e24` | `5b6c8a58d9359a939d615abae455548aab8a79b3` | Measure quantized repair in audit space |
| `9af0f39eb5cedebbd13b8d8bd38053cea777e6c0` | `8a49783b0faa302912dea54f47ea0a32ae7481b5` | Align quantized repair with audited ratio |
| `9dd9ba39d41b7196d30666f87307b32f03de32c7` | `46b1d3aba89b5eb32f52fe190eea52412eaaf926` | Document matched-budget diagnostic results |
| `fad012e0be8e340751e47d26b0c20e534e34e3f0` | `a11bfff8e12fcababd30d4c559a19da477730ab5` | Document complete research snapshot |
