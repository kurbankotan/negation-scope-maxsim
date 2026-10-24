| system | scoring | NevIR paired all | NevIR nested | NevIR ties | Perm SS_bal | Perm ties | Fresh SS | Fresh ties |
|---|---|---|---|---|---|---|---|---|
| monot5-3b | previous: torch.bfloat16 | 46.3 | 39.7 | 118 | 12.9 | 7 | -4.2 | 4 |
| monot5-3b | rescored: float32 head on the float32-cast pre-head state | 50.5 | 43.5 | 0 | 12.2 | 0 | -4.6 | 0 |
| monot5-base | previous: torch.bfloat16 | 34.1 | 29.9 | 18 | 6.2 | 0 | -2.5 | 0 |
| monot5-base | rescored: float32 head on the float32-cast pre-head state | 34.5 | 29.3 | 0 | 6.2 | 0 | -2.5 | 0 |
| olmo-2-7b-instruct | previous: torch.bfloat16 | 37.2 | 33.2 | 0 | 73.7 | 0 | 63.3 | 0 |
| olmo-2-7b-instruct | rescored: float32 head on the float32-cast pre-head state | 37.5 | 34.2 | 0 | 73.7 | 0 | 63.3 | 0 |
| phi-4-mini-instruct | previous: torch.bfloat16 | 59.4 | 47.8 | 11 | 77.6 | 0 | 77.1 | 0 |
| phi-4-mini-instruct | rescored: float32 head on the float32-cast pre-head state | 59.4 | 48.9 | 0 | 77.7 | 0 | 77.1 | 0 |
| qwen3-8b | previous: torch.bfloat16 | 56.7 | 50.5 | 26 | 90.1 | 0 | 84.6 | 3 |
| qwen3-8b | rescored: float32 head on the float32-cast pre-head state | 57.8 | 53.3 | 0 | 90.5 | 0 | 85.4 | 0 |