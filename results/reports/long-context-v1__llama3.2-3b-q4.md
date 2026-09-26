# Long-Context V1 Report

configured 4096-token window; no native 8K/32K claim

| Length | Passes | Trials | Accuracy |
|---|---:|---:|---:|
| L | 0 | 21 | 0.000 |
| M | 9 | 21 | 0.429 |
| S | 9 | 21 | 0.429 |

| Archetype | Passes | Trials | Accuracy |
|---|---:|---:|---:|
| end_needle | 0 | 9 | 0.000 |
| global_aggregate | 6 | 9 | 0.667 |
| grounded_abstention | 6 | 9 | 0.667 |
| middle_needle | 0 | 9 | 0.000 |
| multi_needle | 6 | 9 | 0.667 |
| start_needle | 0 | 9 | 0.000 |
| state_recency | 0 | 9 | 0.000 |
