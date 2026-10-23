| file | size GB | engine | HumanEval | HE+ | KLD vs F16 | same top-p % | PPL |
|---|---|---|---|---|---|---|---|
| Qwen3.5-4B F16 | 8.67 | llama.cpp | 57.93 | 53.05 | — | — | 9.73 |
| Q4_K_M | 2.78 | llama.cpp | 62.80 | 54.27 | 0.0271 | 92.4 | 9.87 |
| Q4_K_M (opencoti engine control) | 2.78 | opencoti | 60.37 | 55.49 | 0.0272 | 92.6 | 9.86 |
| Q4_K_S | 2.63 | llama.cpp | 59.76 | 53.66 | 0.0307 | 91.7 | 9.85 |
| Q3_K_L | 2.48 | llama.cpp | 54.27 | 49.39 | 0.0874 | 86.2 | 10.48 |
| Q3_K_M | 2.32 | llama.cpp | 51.83 | 48.78 | 0.0928 | 85.6 | 10.41 |
| Q3_K_S | 2.12 | llama.cpp | 49.39 | 43.29 | 0.1324 | 83.1 | 10.82 |
| IQ3_M | 2.22 | llama.cpp | 55.49 | 48.78 | 0.0909 | 86.2 | 10.39 |
| IQ3_XS | 2.15 | llama.cpp | 49.39 | 43.90 | 0.1122 | 84.5 | 10.70 |
| IQ3_XXS | 1.97 | llama.cpp | 48.17 | 42.68 | 0.1511 | 82.0 | 11.04 |
| student IQ2_XXS core, IQ3_M packaging + Q8_0 head | 1.98 | llama.cpp | 29.27 | 25.00 | 0.3755 | 73.4 | 12.49 |
| student IQ2_XXS core, Q3_K_M packaging | 1.83 | llama.cpp | 32.32 | 28.05 | 0.3668 | 74.6 | 12.73 |
| student IQ2_XXS core, IQ3_M packaging | 1.83 | llama.cpp | 30.49 | 26.22 | 0.3654 | 74.4 | 12.56 |
| IQ2_M | 1.81 | llama.cpp | 32.32 | 27.44 | 0.2818 | 75.7 | 12.04 |
| student MLP re-encoded at IQ2_XS (off its trained grid) | 1.73 | llama.cpp | 28.05 | 23.17 | 0.3240 | 75.7 | 12.41 |
| IQ2_S | 1.72 | llama.cpp | 23.17 | 18.90 | 0.4019 | 71.6 | 14.11 |
| IQ2_XS | 1.70 | llama.cpp | 20.12 | 18.29 | 0.4391 | 70.4 | 14.38 |
| student IQ2_XXS core, IQ3_XXS packaging | 1.69 | llama.cpp | 36.59 | 31.71 | 0.3632 | 74.7 | 12.64 |
| **student IQ2_XXS stage 2 (mix_v2, lr 3e-6)** | 1.66 | llama.cpp | 42.07 | 39.02 | 0.3235 | 76.9 | 12.20 |
| **student IQ2_XXS (as trained, IQ2_M packaging)** | 1.66 | llama.cpp | 38.41 | 31.71 | 0.3503 | 75.7 | 12.48 |
| IQ2_XXS | 1.61 | llama.cpp | 10.37 | 9.76 | 0.6181 | 65.6 | 16.73 |
| student MLP re-encoded at IQ1_M (off its trained grid) | 1.57 | llama.cpp | 14.02 | 12.20 | 0.6283 | 67.1 | 16.45 |
| IQ1_M | 1.49 | llama.cpp | 2.44 | 1.83 | 1.2275 | 53.8 | 30.36 |
| **student ternary (PQ2_0)** | 1.20 | opencoti | 17.68 | 15.85 | 0.9117 | 58.6 | 22.42 |
