# SPDX-License-Identifier: Apache-2.0
"""IQ2_XXS / IQ2_XS / IQ2_S as native QAD formats: a CUDA encoder that reproduces llama.cpp's, and their decoder.

The encoder is llama.cpp's ``quantize_row_iq2_xxs_impl`` / ``quantize_row_iq2_xs_impl`` / ``quantize_row_iq2_s_impl``
(ggml-quants.c) ported operation by operation, with the lattice tables built exactly as ``iq2xs_init_impl`` builds
them (bm-896: ModelOpt's IQ2_XXS encoder scored 31 % worse than llama-quantize's on gemma-4-26B, so IQ2 went through
libggml on the CPU; bm-900, 2026-11-19: ModelOpt's IQ2_XS produced none of llama-quantize's blocks either):

  * sub-blocks of 32 (IQ2_XXS) / 16 (IQ2_XS, IQ2_S) values taken by magnitude with per-8 sign bits (IQ2_XXS and
    IQ2_XS: an even number of flips, the least important value flipped back if needed, 7 stored bits + parity;
    IQ2_S: 8 stored bits);
  * weights ``qw * sqrt(sigma2 + x^2)`` with ``sigma2 = sum x^2 / 256`` (IQ2_XXS, IQ2_XS: imatrix mandatory) or
    ``2 sum x^2 / 256`` (IQ2_S; without an imatrix ``0.25 sigma2 + x^2``);
  * 8-value groups snapped to a 256 (IQ2_XXS) / 512 (IQ2_XS) / 1024 (IQ2_S) point grid of odd levels 1 / 3 / 5: a
    group off the grid takes the best of its precomputed neighbours (2 / 2 / 1 nearest distance shells) under
    ``sqrt(weight)``;
  * IQ2_XXS: a first scale from ``make_qp_quants`` (4 levels, 8 trial scales, 5 greedy passes), then 13 trial scales
    ``(5 + 0.1 t) / (3 scale)``; IQ2_XS and IQ2_S: 19 trial scales ``(5 + 0.1 t) / max``. The best weighted fit wins,
    then a re-snap at the winning scale (IQ2_XXS: every group; IQ2_XS, IQ2_S: the off-grid groups) and a final
    least-squares scale;
  * 256-value super-block: fp16 ``d = max_scale / 31`` (IQ2_S times 0.9875) and 4-bit sub-block scales.

Every sum runs in llama.cpp's order in float32 without fused multiply-adds (``--fmad=false`` plus explicit
round-to-nearest intrinsics), so the bytes match libggml's bit for bit. On a CUDA device encoding ALWAYS uses the
CUDA kernel (no CPU fallback; an extension that cannot be built raises); :func:`encode_reference` is the same
algorithm in torch, for CPU devices and as the kernel's oracle; libggml (``omk_qad.ggml_lib``) is the oracle of both.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import torch

KGRID_2BIT_256 = (  # kgrid_2bit_256 (ggml-quants.c iq2xs_init_impl): 256 points, 8 x 2-bit levels per point
    0, 2, 5, 8, 10, 17, 20, 32, 34, 40, 42, 65, 68, 80, 88, 97,
    100, 128, 130, 138, 162, 257, 260, 272, 277, 320, 388, 408, 512, 514, 546, 642,
    1025, 1028, 1040, 1057, 1060, 1088, 1090, 1096, 1120, 1153, 1156, 1168, 1188, 1280, 1282, 1288,
    1312, 1350, 1385, 1408, 1425, 1545, 1552, 1600, 1668, 1700, 2048, 2053, 2056, 2068, 2088, 2113,
    2116, 2128, 2130, 2184, 2308, 2368, 2562, 2580, 4097, 4100, 4112, 4129, 4160, 4192, 4228, 4240,
    4245, 4352, 4360, 4384, 4432, 4442, 4480, 4644, 4677, 5120, 5128, 5152, 5157, 5193, 5248, 5400,
    5474, 5632, 5654, 6145, 6148, 6160, 6208, 6273, 6400, 6405, 6560, 6737, 8192, 8194, 8202, 8260,
    8289, 8320, 8322, 8489, 8520, 8704, 8706, 9217, 9220, 9232, 9280, 9302, 9472, 9537, 9572, 9872,
    10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
    16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
    17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
    20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
    22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
    25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
    33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
    37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
)

KGRID_2BIT_512 = (  # kgrid_2bit_512 (ggml-quants.c iq2xs_init_impl): 512 points, 8 x 2-bit levels per point
    0, 2, 5, 8, 10, 17, 20, 22, 25, 32, 34, 37, 40, 65, 68, 70,
    73, 80, 82, 85, 88, 97, 100, 128, 130, 133, 136, 145, 148, 153, 160, 257,
    260, 262, 265, 272, 274, 277, 280, 282, 289, 292, 320, 322, 325, 328, 337, 340,
    352, 360, 385, 388, 400, 512, 514, 517, 520, 529, 532, 544, 577, 580, 592, 597,
    640, 650, 1025, 1028, 1030, 1033, 1040, 1042, 1045, 1048, 1057, 1060, 1088, 1090, 1093, 1096,
    1105, 1108, 1110, 1120, 1153, 1156, 1168, 1280, 1282, 1285, 1288, 1297, 1300, 1312, 1345, 1348,
    1360, 1377, 1408, 1537, 1540, 1552, 1574, 1600, 1602, 1668, 2048, 2050, 2053, 2056, 2058, 2065,
    2068, 2080, 2085, 2113, 2116, 2128, 2136, 2176, 2208, 2218, 2305, 2308, 2320, 2368, 2433, 2441,
    2560, 2592, 2600, 2710, 2720, 4097, 4100, 4102, 4105, 4112, 4114, 4117, 4120, 4129, 4132, 4160,
    4162, 4165, 4168, 4177, 4180, 4192, 4202, 4225, 4228, 4240, 4352, 4354, 4357, 4360, 4369, 4372,
    4384, 4417, 4420, 4432, 4480, 4500, 4502, 4609, 4612, 4614, 4624, 4672, 4704, 5120, 5122, 5125,
    5128, 5137, 5140, 5152, 5185, 5188, 5193, 5200, 5220, 5248, 5377, 5380, 5392, 5440, 5632, 5652,
    5705, 6145, 6148, 6160, 6162, 6208, 6228, 6278, 6400, 6405, 6502, 6737, 6825, 8192, 8194, 8197,
    8200, 8202, 8209, 8212, 8224, 8257, 8260, 8272, 8320, 8352, 8449, 8452, 8464, 8512, 8520, 8549,
    8704, 8738, 8832, 8872, 9217, 9220, 9232, 9257, 9280, 9472, 9537, 9554, 9625, 9729, 9754, 9894,
    10240, 10248, 10250, 10272, 10325, 10376, 10402, 10600, 10640, 10760, 10784, 10882, 10888, 10890, 16385, 16388,
    16390, 16393, 16400, 16402, 16405, 16408, 16417, 16420, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16480,
    16485, 16513, 16516, 16528, 16640, 16642, 16645, 16648, 16657, 16660, 16672, 16705, 16708, 16720, 16768, 16773,
    16802, 16897, 16900, 16912, 16914, 16937, 16960, 17408, 17410, 17413, 17416, 17425, 17428, 17433, 17440, 17473,
    17476, 17488, 17536, 17556, 17665, 17668, 17680, 17700, 17728, 17818, 17920, 17930, 17988, 18000, 18433, 18436,
    18448, 18496, 18501, 18516, 18530, 18688, 18705, 18756, 18768, 18793, 18948, 20480, 20482, 20485, 20488, 20497,
    20500, 20512, 20520, 20545, 20548, 20560, 20608, 20737, 20740, 20752, 20757, 20800, 20802, 20992, 21060, 21162,
    21505, 21508, 21520, 21537, 21568, 21600, 21633, 21665, 21760, 21768, 21888, 21896, 22049, 22120, 22177, 22528,
    22548, 22593, 22608, 22681, 22810, 22848, 22850, 23173, 24577, 24580, 24592, 24640, 24660, 24674, 24710, 24745,
    24832, 25124, 25162, 25234, 25600, 25622, 25872, 25920, 25925, 26020, 26625, 26730, 26917, 27142, 27220, 27234,
    32768, 32770, 32773, 32776, 32785, 32788, 32800, 32810, 32833, 32836, 32848, 32896, 32898, 32936, 32938, 33025,
    33028, 33030, 33040, 33088, 33105, 33113, 33280, 33312, 33408, 33410, 33440, 33448, 33793, 33796, 33808, 33810,
    33813, 33856, 33888, 33929, 34048, 34116, 34213, 34328, 34410, 34816, 34824, 34853, 34906, 34944, 34946, 34984,
    35078, 35362, 35456, 35464, 35478, 35496, 36865, 36868, 36880, 36928, 36950, 36996, 37120, 37154, 37220, 37462,
    37513, 37888, 37893, 37956, 37968, 37976, 38185, 38288, 38290, 38465, 38993, 39078, 39241, 39445, 39520, 40960,
    40962, 40968, 40970, 40992, 41002, 41120, 41297, 41305, 41382, 41472, 41474, 41480, 41514, 41600, 41632, 42048,
    42133, 42597, 42648, 43018, 43040, 43042, 43048, 43168, 43176, 43268, 43396, 43398, 43560, 43562, 43665, 43690,
)

KGRID_2BIT_1024 = (  # kgrid_2bit_1024 (ggml-quants.c iq2xs_init_impl): 1024 points, 8 x 2-bit levels per point
    0, 2, 5, 8, 10, 17, 20, 22, 25, 32, 34, 37, 40, 65, 68, 70,
    73, 80, 82, 85, 88, 97, 100, 102, 105, 128, 130, 133, 136, 145, 148, 160,
    165, 170, 257, 260, 262, 265, 272, 274, 277, 280, 289, 292, 320, 322, 325, 328,
    337, 340, 342, 345, 352, 357, 360, 385, 388, 400, 402, 405, 417, 420, 512, 514,
    517, 520, 529, 532, 544, 554, 577, 580, 582, 585, 592, 597, 640, 645, 650, 660,
    674, 1025, 1028, 1030, 1033, 1040, 1042, 1045, 1048, 1057, 1060, 1062, 1065, 1088, 1090, 1093,
    1096, 1098, 1105, 1108, 1110, 1113, 1120, 1122, 1125, 1153, 1156, 1158, 1161, 1168, 1173, 1176,
    1185, 1188, 1280, 1282, 1285, 1288, 1290, 1297, 1300, 1302, 1305, 1312, 1317, 1320, 1345, 1348,
    1350, 1353, 1360, 1362, 1365, 1368, 1377, 1380, 1408, 1410, 1413, 1416, 1425, 1428, 1440, 1537,
    1540, 1542, 1545, 1552, 1557, 1600, 1605, 1608, 1617, 1620, 1632, 1665, 1668, 1680, 2048, 2050,
    2053, 2056, 2065, 2068, 2070, 2073, 2080, 2085, 2090, 2113, 2116, 2118, 2121, 2128, 2130, 2133,
    2136, 2145, 2148, 2176, 2181, 2196, 2218, 2305, 2308, 2320, 2322, 2325, 2328, 2337, 2368, 2373,
    2376, 2385, 2388, 2400, 2433, 2448, 2560, 2577, 2580, 2594, 2600, 2602, 2640, 2713, 4097, 4100,
    4102, 4105, 4112, 4114, 4117, 4120, 4129, 4132, 4134, 4160, 4162, 4165, 4168, 4177, 4180, 4182,
    4185, 4192, 4194, 4197, 4200, 4225, 4228, 4230, 4240, 4245, 4248, 4257, 4260, 4352, 4354, 4357,
    4360, 4362, 4369, 4372, 4374, 4377, 4384, 4386, 4389, 4392, 4417, 4420, 4422, 4425, 4432, 4434,
    4437, 4440, 4449, 4452, 4480, 4482, 4485, 4488, 4497, 4500, 4609, 4612, 4617, 4624, 4629, 4641,
    4644, 4672, 4677, 4689, 4692, 4737, 4740, 4752, 5120, 5122, 5125, 5128, 5137, 5140, 5142, 5145,
    5152, 5157, 5160, 5185, 5188, 5190, 5193, 5200, 5202, 5205, 5208, 5217, 5220, 5248, 5250, 5253,
    5256, 5265, 5268, 5280, 5377, 5380, 5382, 5385, 5392, 5394, 5397, 5400, 5409, 5412, 5440, 5442,
    5445, 5448, 5457, 5460, 5472, 5505, 5508, 5520, 5632, 5637, 5640, 5649, 5652, 5664, 5697, 5700,
    5712, 5760, 5802, 6145, 6148, 6150, 6153, 6160, 6165, 6168, 6177, 6208, 6210, 6213, 6216, 6225,
    6228, 6240, 6273, 6276, 6400, 6402, 6405, 6408, 6417, 6420, 6432, 6465, 6468, 6480, 6505, 6562,
    6660, 6672, 6720, 6742, 8192, 8194, 8197, 8200, 8209, 8212, 8214, 8217, 8224, 8229, 8234, 8257,
    8260, 8272, 8274, 8277, 8292, 8320, 8330, 8340, 8362, 8449, 8452, 8464, 8466, 8469, 8481, 8512,
    8514, 8517, 8529, 8532, 8544, 8577, 8580, 8592, 8704, 8714, 8738, 8744, 8746, 8772, 8784, 8840,
    8842, 8872, 9217, 9220, 9222, 9225, 9232, 9237, 9240, 9249, 9252, 9280, 9282, 9285, 9288, 9297,
    9300, 9312, 9345, 9348, 9360, 9472, 9477, 9480, 9489, 9492, 9504, 9537, 9540, 9552, 9574, 9600,
    9729, 9732, 9744, 9792, 9817, 10240, 10245, 10257, 10260, 10305, 10308, 10320, 10378, 10410, 10497, 10500,
    10512, 10645, 10762, 10786, 10852, 10888, 10890, 16385, 16388, 16390, 16393, 16400, 16402, 16405, 16408, 16410,
    16417, 16420, 16422, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16470, 16473, 16480, 16482, 16485, 16513,
    16516, 16528, 16533, 16536, 16545, 16548, 16640, 16642, 16645, 16648, 16657, 16660, 16662, 16665, 16672, 16674,
    16677, 16705, 16708, 16710, 16713, 16720, 16722, 16725, 16728, 16737, 16740, 16768, 16770, 16773, 16776, 16785,
    16788, 16800, 16897, 16900, 16912, 16914, 16917, 16920, 16932, 16960, 16965, 16968, 16977, 16980, 16992, 17025,
    17028, 17408, 17410, 17413, 17416, 17418, 17425, 17428, 17430, 17433, 17440, 17442, 17445, 17448, 17473, 17476,
    17478, 17481, 17488, 17490, 17493, 17496, 17505, 17508, 17536, 17538, 17541, 17544, 17553, 17556, 17568, 17665,
    17668, 17670, 17673, 17680, 17682, 17685, 17688, 17697, 17700, 17728, 17730, 17733, 17736, 17745, 17748, 17760,
    17770, 17793, 17796, 17808, 17920, 17922, 17925, 17928, 17937, 17940, 17952, 17985, 17988, 18000, 18048, 18085,
    18433, 18436, 18441, 18448, 18450, 18453, 18456, 18465, 18468, 18496, 18498, 18501, 18504, 18513, 18516, 18528,
    18564, 18576, 18688, 18690, 18693, 18696, 18705, 18708, 18720, 18753, 18756, 18768, 18816, 18838, 18945, 18948,
    18960, 19008, 20480, 20482, 20485, 20488, 20497, 20500, 20502, 20505, 20512, 20514, 20517, 20520, 20545, 20548,
    20550, 20553, 20560, 20562, 20565, 20568, 20577, 20580, 20608, 20610, 20613, 20616, 20625, 20628, 20737, 20740,
    20742, 20745, 20752, 20754, 20757, 20760, 20769, 20772, 20800, 20802, 20805, 20808, 20817, 20820, 20832, 20865,
    20868, 20880, 20992, 20997, 21000, 21009, 21012, 21024, 21057, 21060, 21072, 21097, 21120, 21505, 21508, 21510,
    21513, 21520, 21522, 21525, 21528, 21537, 21540, 21568, 21570, 21573, 21576, 21585, 21588, 21600, 21633, 21636,
    21648, 21760, 21762, 21765, 21768, 21777, 21780, 21792, 21825, 21828, 21840, 21888, 22017, 22020, 22032, 22054,
    22080, 22528, 22530, 22533, 22536, 22545, 22548, 22560, 22593, 22596, 22608, 22618, 22656, 22785, 22788, 22800,
    22848, 23040, 23065, 23173, 23208, 24577, 24580, 24582, 24592, 24594, 24597, 24600, 24609, 24612, 24640, 24645,
    24648, 24657, 24660, 24672, 24708, 24720, 24832, 24834, 24837, 24840, 24849, 24852, 24864, 24897, 24900, 24912,
    24960, 24985, 25092, 25104, 25152, 25174, 25249, 25600, 25605, 25608, 25617, 25620, 25632, 25665, 25668, 25680,
    25728, 25857, 25860, 25872, 25920, 25930, 25960, 26002, 26112, 26260, 26625, 26628, 26640, 26725, 26776, 26880,
    26922, 27202, 27297, 32768, 32770, 32773, 32776, 32785, 32788, 32793, 32800, 32805, 32833, 32836, 32848, 32850,
    32853, 32856, 32865, 32896, 32901, 32913, 32916, 33025, 33028, 33033, 33040, 33042, 33045, 33048, 33057, 33060,
    33088, 33090, 33093, 33096, 33105, 33108, 33153, 33156, 33168, 33193, 33280, 33285, 33290, 33297, 33300, 33345,
    33348, 33360, 33793, 33796, 33798, 33801, 33808, 33810, 33813, 33816, 33825, 33856, 33858, 33861, 33864, 33873,
    33876, 33888, 33921, 33924, 33936, 34048, 34050, 34053, 34056, 34065, 34068, 34080, 34113, 34116, 34128, 34176,
    34186, 34305, 34308, 34320, 34345, 34368, 34816, 34821, 34833, 34836, 34881, 34884, 34896, 34978, 35073, 35076,
    35136, 35173, 35362, 35416, 35418, 35458, 35490, 36865, 36868, 36873, 36880, 36882, 36885, 36888, 36900, 36928,
    36930, 36933, 36936, 36945, 36948, 36960, 36993, 36996, 37008, 37120, 37125, 37137, 37140, 37185, 37188, 37200,
    37210, 37377, 37380, 37392, 37440, 37542, 37888, 37890, 37893, 37896, 37905, 37908, 37920, 37953, 37956, 37968,
    38016, 38038, 38145, 38148, 38160, 38208, 38296, 38305, 38400, 38470, 38500, 38913, 38916, 38928, 38950, 38976,
    39081, 39168, 39241, 39250, 39568, 40960, 40965, 40970, 40980, 40994, 41002, 41025, 41028, 41040, 41122, 41130,
    41280, 41317, 41474, 41482, 41506, 41512, 41514, 41602, 41608, 41610, 41640, 41985, 41988, 42000, 42048, 42121,
    42148, 42240, 42265, 42577, 43018, 43048, 43170, 43348, 43398, 43528, 43530, 43552, 43554, 43560, 43656, 43690,
)

SPECS = {"IQ2_XXS": (256, 66), "IQ2_XS": (256, 74), "IQ2_S": (256, 82)}  # qtype -> (values per block, bytes per block)
TYPES = frozenset(SPECS)
KIND = {"IQ2_XXS": 0, "IQ2_XS": 1, "IQ2_S": 2}  # the kernel's variant index
PARITY = frozenset({"IQ2_XXS", "IQ2_XS"})  # 7 sign bits + parity (an even number of flips); imatrix mandatory
KMAXQ = 3  # levels 0..2 -> odd values 1 / 3 / 5
KMAP = 43692  # iq2xs_init_impl's kmap_size: codes of 8 x 2-bit levels <= 2 (0xAAAA + 2)
EPS = {"IQ2_XXS": np.float32(1e-15), "IQ2_XS": np.float32(1e-15), "IQ2_S": np.float32(1e-8)}  # GROUP_MAX_EPS[_IQ2_S]
_CFG = {  # grid points, distance shells kept per off-grid point, sub-block, trial scales, fp16 fudge
    "IQ2_XXS": (256, 2, 32, range(-6, 7), np.float32(1.0)),
    "IQ2_XS": (512, 2, 16, range(-9, 10), np.float32(1.0)),
    "IQ2_S": (1024, 1, 16, range(-9, 10), np.float32(0.9875)),
}
_KGRID = {256: KGRID_2BIT_256, 512: KGRID_2BIT_512, 1024: KGRID_2BIT_1024}
_DEC = np.array([8, 25, 43], dtype=np.float32)  # iq2xxs/iq2xs/iq2s_grid bytes per level (dequantize scale 0.25)


@cache
def tables(qtype: str) -> dict[str, np.ndarray]:
    """iq2xs_init_impl: grid (odd levels), kmap (index or -(neighbour offset + 1)), neighbour lists, decoder grid."""
    n, nwant = _CFG[qtype][:2]
    kgrid = _KGRID[n]
    grid = np.array([[2 * ((g >> 2 * i) & 3) + 1 for i in range(8)] for g in kgrid], dtype=np.int32)  # [n, 8]
    kmap = np.full(KMAP, -1, dtype=np.int32)
    kmap[[sum(((v - 1) // 2) << 2 * i for i, v in enumerate(p)) for p in grid]] = np.arange(n, dtype=np.int32)
    neighbours: list[int] = []
    lists = {}
    for u in range(KMAP):
        if kmap[u] >= 0:
            continue
        pos = np.array([2 * ((u >> 2 * k) & 3) + 1 for k in range(8)], dtype=np.int32)
        d2 = ((grid - pos) ** 2).sum(1)
        order = np.lexsort((np.arange(n), d2))  # qsort by (d2, index)
        take, cur, nhave = [], d2[order[0]], 1
        for j in order:
            if d2[j] > cur:
                if nhave == nwant:
                    break
                cur, nhave = d2[j], nhave + 1
            take.append(int(j))
        kmap[u] = -(len(neighbours) + 1)
        neighbours += [len(take), *take]
        lists[u] = take
    pad = np.full((KMAP, max(len(t) for t in lists.values())), -1, dtype=np.int64)
    for u, t in lists.items():
        pad[u, : len(t)] = t
    signs = np.array([s | ((s.bit_count() & 1) << 7) for s in range(128)], dtype=np.int64)  # ksigns_iq2xs
    return {"grid": grid, "kmap": kmap, "neighbours": np.array(neighbours, dtype=np.int32), "pad": pad,
            "dec": _DEC[(grid - 1) // 2], "ksigns": signs}


def _dev_tables(qtype: str, device: torch.device) -> dict[str, torch.Tensor]:
    key = (qtype, str(device))
    cache_ = _dev_tables.__dict__.setdefault("cache", {})
    if key not in cache_:
        t = tables(qtype)
        cache_[key] = {
            "grid": torch.from_numpy(t["grid"].astype(np.float32)).to(device),
            "gridl": torch.from_numpy((t["grid"].astype(np.int64) - 1) // 2).to(device),
            "kmap": torch.from_numpy(t["kmap"].astype(np.int64)).to(device),
            "pad": torch.from_numpy(t["pad"]).to(device),
            "dec": torch.from_numpy(t["dec"]).to(device),
            "ksigns": torch.from_numpy(t["ksigns"]).to(device),
            # kernel inputs: uint8 [n, 8] grid of odd levels, int32 kmap, int32 neighbour lists
            "grid_u8": torch.from_numpy(t["grid"].astype(np.uint8).reshape(-1)).to(device),
            "kmap_i32": torch.from_numpy(t["kmap"]).to(device),
            "neigh_i32": torch.from_numpy(t["neighbours"]).to(device),
        }
    return cache_[key]


def _ext():
    """The CUDA encoder (built on first use); raises if it cannot be built -- there is no CPU fallback."""
    if not hasattr(_ext, "mod"):
        from modelopt.torch.utils.cpp_extension import load_cpp_extension

        here = Path(__file__).parent / "kernels"
        _ext.mod = load_cpp_extension(
            name="omk_cuda_ext_iq2",
            sources=[here / "iq2.cpp", here / "iq2.cu"],
            cuda_version_specifiers=">=11.8",
            fail_msg="omk_qad IQ2_XXS/IQ2_XS/IQ2_S CUDA encoder is unavailable (no CPU fallback on a CUDA device).",
            extra_cuda_cflags=["-O3", "--fmad=false"],
            raise_if_failed=True,
        )
    return _ext.mod


def _check(qtype: str, x: torch.Tensor, importance: torch.Tensor | None) -> None:
    if qtype not in SPECS:
        raise ValueError(f"not an IQ2 type: {qtype}")
    if x.dim() != 2 or x.dtype != torch.float32 or x.shape[1] % 256:
        raise ValueError(f"{qtype} needs float32 [rows, cols] with cols a multiple of 256, got {x.dtype} "
                         f"{tuple(x.shape)}")
    if importance is None and qtype in PARITY:
        raise ValueError(f"{qtype} requires an importance matrix (llama.cpp: 'missing quantization weights')")
    if importance is not None and (importance.dim() != 1 or importance.numel() != x.shape[1]):
        raise ValueError(f"importance must be [cols={x.shape[1]}], got {tuple(importance.shape)}")


def encode(qtype: str, x: torch.Tensor, importance: torch.Tensor | None = None) -> torch.Tensor:
    """uint8 payload [rows, cols / 256, block bytes] of float32 ``x`` [rows, cols]; ``importance`` = E[x^2] [cols]."""
    _check(qtype, x, importance)
    if not x.is_cuda:
        return encode_reference(qtype, x, importance)
    t = _dev_tables(qtype, x.device)
    imp = None if importance is None else importance.to(device=x.device, dtype=torch.float32).contiguous()
    return _ext().iq2_pack(x.contiguous(), imp, KIND[qtype], t["grid_u8"], t["kmap_i32"], t["neigh_i32"])


def decode(qtype: str, packed: torch.Tensor, rows: int, cols: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[rows, cols] values of a payload (llama.cpp's dequantize_row_iq2_xxs / _xs / _s), on the payload's device."""
    nbytes = SPECS[qtype][1]
    p = packed.reshape(-1, nbytes)
    t = _dev_tables(qtype, p.device)
    d = p[:, :2].contiguous().view(torch.float16).float()  # [B, 1]
    bit = torch.arange(8, device=p.device)
    if qtype == "IQ2_XXS":
        q = p[:, 2:66].contiguous().view(torch.int32).long().reshape(-1, 8, 2) & 0xFFFFFFFF  # [B, ib, (grid, signs)]
        gi = (q[:, :, 0, None] >> (8 * torch.arange(4, device=p.device))) & 0xFF  # [B, 8, 4]
        db = (d * ((q[:, :, 1] >> 28).float() + 0.5)) * 0.25  # [B, 8]
        sg = t["ksigns"][(q[:, :, 1, None] >> (7 * torch.arange(4, device=p.device))) & 127]  # [B, 8, 4]
    elif qtype == "IQ2_XS":
        q = p[:, 2:66].contiguous().view(torch.int16).long().reshape(-1, 8, 4) & 0xFFFF  # qs[4 ib32 + l]: 9 + 7 bits
        gi = q & 511
        sg = t["ksigns"][q >> 9]  # [B, 8, 4]
        sc = p[:, 66:74].long()  # [B, 8]: low nibble -> l 0..1, high nibble -> l 2..3
        db = torch.stack([(d * ((sc & 0xF).float() + 0.5)) * 0.25, (d * ((sc >> 4).float() + 0.5)) * 0.25], 2)
        db = db.repeat_interleave(2, 2)  # [B, 8, 4]
    else:
        qs = p[:, 2:34].long().reshape(-1, 8, 4)  # [B, ib32, l]
        qh = p[:, 66:74].long()  # [B, 8]
        gi = qs | ((qh[:, :, None] << (8 - 2 * torch.arange(4, device=p.device))) & 0x300)
        sg = p[:, 34:66].long().reshape(-1, 8, 4)
        sc = p[:, 74:82].long()  # [B, 8]: low nibble -> l 0..1, high nibble -> l 2..3
        db = torch.stack([(d * ((sc & 0xF).float() + 0.5)) * 0.25, (d * ((sc >> 4).float() + 0.5)) * 0.25], 2)
        db = db.repeat_interleave(2, 2)  # [B, 8, 4]
    vals = t["dec"][gi]  # [B, 8, 4, 8]
    neg = (sg[..., None] >> bit) & 1
    dbv = db[:, :, :, None] if db.dim() == 3 else db[:, :, None, None]
    y = (dbv * vals) * (1 - 2 * neg).float()
    return y.reshape(rows, cols).to(dtype)


# ------------------------------------------------------------------------------------------- torch reference
def _f32(v, like: torch.Tensor) -> torch.Tensor:
    return torch.full_like(like, float(np.float32(v)))


def _nearest(v: torch.Tensor) -> torch.Tensor:
    """nearest_int: round half to even (the 12582912 trick), as float32."""
    return torch.round(v)


def _snap(t: dict, u: torch.Tensor, xv: torch.Tensor, wa: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """iq2_find_best_neighbour for groups whose code u is off the grid: grid index of the best neighbour."""
    cand = t["pad"][u]  # [G, maxn]
    q = t["grid"][cand.clamp(min=0)]  # [G, maxn, 8]
    d2 = torch.zeros(cand.shape, dtype=torch.float32, device=u.device)
    for i in range(8):
        diff = scale[:, None] * q[:, :, i] - xv[:, None, i]
        d2 = d2 + (wa[:, None, i] * diff) * diff
    d2 = torch.where(cand >= 0, d2, torch.full_like(d2, float("inf")))
    return cand.gather(1, d2.argmin(1, keepdim=True))[:, 0]  # first minimum (strict <)


def _levels(t: dict, inv: torch.Tensor, xval: torch.Tensor, wa: torch.Tensor, scale: torch.Tensor, ng: int):
    """Levels [S, 8 ng] at inverse scale ``inv`` [S] with off-grid groups snapped at ``scale``; on-grid mask [S, ng]."""
    lev = _nearest((inv[:, None] * xval - 1) * 0.5).clamp(0, KMAXQ - 1).long()  # nearest_int(0.5f*(id*x - 1))
    l8 = lev.reshape(-1, ng, 8)
    u = (l8 << (2 * torch.arange(8, device=lev.device))).sum(-1)  # [S, ng]
    on = t["kmap"][u] >= 0
    off = (~on).nonzero(as_tuple=True)
    if off[0].numel():
        g = _snap(t, u[off], xval.reshape(-1, ng, 8)[off], wa.reshape(-1, ng, 8)[off], scale[off[0]])
        l8 = l8.clone()
        l8[off] = t["gridl"][g]
    return l8.reshape(-1, 8 * ng), on


def _sums(w: torch.Tensor, xval: torch.Tensor, lev: torch.Tensor):
    q = (2 * lev + 1).float()
    sumqx = torch.zeros(w.shape[0], dtype=torch.float32, device=w.device)
    sumq2 = torch.zeros_like(sumqx)
    for i in range(w.shape[1]):
        sumqx = sumqx + (w[:, i] * xval[:, i]) * q[:, i]
        sumq2 = sumq2 + (w[:, i] * q[:, i]) * q[:, i]
    return sumqx, sumq2


def _make_qp_quants(x: torch.Tensor, w: torch.Tensor, nmax: int = 4) -> torch.Tensor:
    """make_qp_quants(32, 4, xval, L, weight): the first scale of an IQ2_XXS sub-block, per row of x [S, 32].

    llama.cpp keeps ``L`` in a uint8_t array: a NEGATIVE level (the parity flip leaves one value of an odd group
    negative, and ``nearest_int(iscale * x)`` of it can be -1) wraps to 255 wherever ``L[i]`` is read back, while
    the ``int l`` temporaries stay negative. Reproduced here: ``lu`` holds the stored bytes, ``l`` the ints.
    """
    n = x.shape[1]
    mx = x.amax(1)
    zero = mx < _f32(1e-15, mx)  # GROUP_MAX_EPS: all zero -> 0 (the caller skipped these already)
    mxs = torch.where(zero, torch.ones_like(mx), mx)
    iscale = _f32(nmax, mx) / mxs
    lu = (_nearest(iscale[:, None] * x).long() & 0xFF).float()  # L[i] = nearest_int(iscale * x[i]) as uint8_t
    scale = torch.ones_like(iscale) / iscale
    best = torch.zeros_like(mx)
    for i in range(n):
        diff = x[:, i] - scale * lu[:, i]
        best = best + (w[:, i] * diff) * diff
    for it in (-4, -3, -2, -1, 1, 2, 3, 4):
        isc = (_f32(np.float32(0.1) * np.float32(it), mx) + _f32(nmax, mx)) / mxs
        sc = torch.ones_like(isc) / isc
        mse = torch.zeros_like(mx)
        for i in range(n):
            l_ = _nearest(isc * x[:, i]).clamp(max=nmax)  # int l = MIN(nmax, nearest_int(...)): may be negative
            diff = x[:, i] - sc * l_
            mse = mse + (w[:, i] * diff) * diff
        better = mse < best
        best = torch.where(better, mse, best)
        iscale = torch.where(better, isc, iscale)
    l_ = _nearest(iscale[:, None] * x).clamp(max=nmax)  # int l per value
    lu = (l_.long() & 0xFF).float()  # L[i] = l (uint8_t)
    sumlx = torch.zeros_like(mx)
    suml2 = torch.zeros_like(mx)
    for i in range(n):  # the sums use the int l
        sumlx = sumlx + (w[:, i] * x[:, i]) * l_[:, i]
        suml2 = suml2 + (w[:, i] * l_[:, i]) * l_[:, i]
    for _ in range(5):  # a pass with no change leaves the state unchanged, so running all 5 equals the early break
        for i in range(n):
            wi, xi, li = w[:, i], x[:, i], lu[:, i]  # L[i] read back as uint8_t
            slx = sumlx - (wi * xi) * li
            sl2 = suml2 - (wi * li) * li
            ok = (slx > 0) & (sl2 > 0)
            slx_s = torch.where(ok, slx, torch.ones_like(slx))
            new_l = _nearest((xi * sl2) / slx_s).clamp(max=nmax)  # int, may be negative
            slx2 = slx + (wi * xi) * new_l
            sl22 = sl2 + (wi * new_l) * new_l
            take = ok & (new_l != li) & ((slx2 * slx2) * suml2 > (sumlx * sumlx) * sl22)
            lu[:, i] = torch.where(take, (new_l.long() & 0xFF).float(), li)
            sumlx = torch.where(take, slx2, sumlx)
            suml2 = torch.where(take, sl22, suml2)
    out = torch.where(suml2 > 0, sumlx / torch.where(suml2 > 0, suml2, torch.ones_like(suml2)), torch.zeros_like(mx))
    return torch.where(zero, torch.zeros_like(out), out)


def _search(qtype: str, xs: torch.Tensor, w: torch.Tensor):
    """Every sub-block [S, sub]: grid indices [S, ng], scale [S], sign bytes [S, ng], skipped mask [S]."""
    _, _, sub, trials, _ = _CFG[qtype]
    ng = sub // 8
    xxs = qtype == "IQ2_XXS"
    parity = qtype in PARITY
    t = _dev_tables(qtype, xs.device)
    ns = xs.shape[0]
    wa = torch.sqrt(w.double()).float()  # correctly rounded sqrtf
    neg = ~(xs >= 0)
    xval = torch.where(neg, -xs, xs)
    bits = 1 << torch.arange(8, device=xs.device)
    sgn = (neg.reshape(ns, ng, 8).long() * bits).sum(-1)  # [S, ng]
    if parity:  # an even number of flips: flip back the least important value of an odd group
        odd = neg.reshape(ns, ng, 8).sum(-1) % 2 == 1
        imin = ((w * xs) * xs).reshape(ns, ng, 8).argmin(-1)  # first minimum (strict <)
        flat = (torch.arange(ng, device=xs.device) * 8 + imin)[odd]
        rows = torch.arange(ns, device=xs.device)[:, None].expand(ns, ng)[odd]
        xval = xval.clone()
        xval[rows, flat] = -xval[rows, flat]
        sgn = torch.where(odd, sgn ^ (1 << imin), sgn) & 127
    mx = xval.amax(1)
    skip = mx < torch.tensor(EPS[qtype], device=xs.device)
    mxs = torch.where(skip, torch.ones_like(mx), mx)  # keeps divisions finite; skipped rows are masked below
    if xxs:
        scale = _make_qp_quants(xval, w)
        eff_max = scale * _f32(KMAXQ, scale)
        skip = skip | (eff_max <= 0)
        denom = torch.where(skip, torch.ones_like(eff_max), eff_max)
    else:
        scale = mxs / _f32(2 * KMAXQ - 1, mx)
        denom = mxs
    best = torch.zeros_like(mx)
    lev = torch.zeros(ns, sub, dtype=torch.long, device=xs.device)
    on = torch.ones((ns, ng), dtype=torch.bool, device=xs.device)
    for it in trials:
        inv = _f32(np.float32(2 * KMAXQ - 1) + np.float32(it) * np.float32(0.1), mx) / denom
        this = torch.ones_like(inv) / inv
        la, ona = _levels(t, inv, xval, wa, this, ng)
        sumqx, sumq2 = _sums(w, xval, la)
        better = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
        nd = sumqx / torch.where(sumq2 > 0, sumq2, torch.ones_like(sumq2))
        scale = torch.where(better, nd, scale)
        best = torch.where(better, nd * sumqx, best)
        lev = torch.where(better[:, None], la, lev)
        on = torch.where(better[:, None], ona, on)
    need = (scale > 0) & ~skip
    if not xxs:
        need = need & (~on).any(1)
    if need.any():
        sc = torch.where(need, scale, torch.ones_like(scale))
        la, _ = _levels(t, torch.ones_like(sc) / sc, xval, wa, sc, ng)
        redo = need[:, None].expand(-1, ng) if xxs else need[:, None] & ~on  # IQ2_S: only the off-grid groups
        lev = torch.where(redo.repeat_interleave(8, 1), la, lev)
        sumqx, sumq2 = _sums(w, xval, lev)
        scale = torch.where(need & (sumq2 > 0), sumqx / torch.where(sumq2 > 0, sumq2, torch.ones_like(sumq2)), scale)
    flip = scale < 0  # "should never happen": positive scale, flipped signs
    scale = torch.where(flip, -scale, scale)
    sgn = torch.where(flip[:, None], ~sgn & (127 if parity else 255), sgn)
    scale = torch.where(skip, torch.zeros_like(scale), scale)
    lev = torch.where(skip[:, None], torch.zeros_like(lev), lev)
    u = (lev.reshape(ns, ng, 8) << (2 * torch.arange(8, device=xs.device))).sum(-1)
    gi = t["kmap"][u]
    assert bool((gi >= 0).all()), "a group is off the grid (llama.cpp aborts here)"
    return gi, scale, sgn, skip


def _weights(qtype: str, x: torch.Tensor, imp: torch.Tensor | None) -> torch.Tensor:
    """Per-value weights of super-blocks x [B, 256] (imp: [B, 256] importance columns or None)."""
    sigma2 = torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)
    for j in range(256):
        sigma2 = sigma2 + x[:, j] * x[:, j]
    if qtype in PARITY:
        sigma2 = sigma2 / _f32(256, sigma2)
    else:
        sigma2 = (2 * sigma2) / _f32(256, sigma2)
    if imp is None:
        return _f32(0.25, sigma2)[:, None] * sigma2[:, None] + x * x
    return imp * torch.sqrt((sigma2[:, None] + x * x).double()).float()


def encode_reference(qtype: str, x: torch.Tensor, importance: torch.Tensor | None = None) -> torch.Tensor:
    """The CUDA kernel's algorithm in torch (any device), byte-identical to it and to libggml."""
    _check(qtype, x, importance)
    rows, cols = x.shape
    nbytes = SPECS[qtype][1]
    _, _, sub, _, fudge = _CFG[qtype]
    nsub, ng = 256 // sub, sub // 8
    xb = x.reshape(-1, 256)
    nb = xb.shape[0]
    imp = None
    if importance is not None:
        imp = importance.to(device=x.device, dtype=torch.float32).reshape(1, cols // 256, 256)
        imp = imp.expand(rows, -1, -1).reshape(-1, 256)
    w = _weights(qtype, xb, imp)
    gi, scale, sgn, skip = _search(qtype, xb.reshape(-1, sub), w.reshape(-1, sub))
    gi, sgn = gi.reshape(nb, nsub, ng), sgn.reshape(nb, nsub, ng)
    scale, skip = scale.reshape(nb, nsub), skip.reshape(nb, nsub)
    out = torch.zeros(nb, nbytes, dtype=torch.uint8, device=x.device)
    max_scale = scale.amax(1)
    live = max_scale > 0
    ms = torch.where(live, max_scale, torch.ones_like(max_scale))
    d = ms / _f32(31, ms)
    dh = torch.where(live, d * _f32(fudge, d), torch.zeros_like(d))
    out[:, :2] = dh.to(torch.float16).reshape(-1, 1).view(torch.uint8)
    inv = torch.ones_like(d) / d
    lsc = _nearest((inv[:, None] * scale - 1) * 0.5).clamp(0, 15).long()  # [nb, nsub]
    gi = torch.where(skip[:, :, None], torch.zeros_like(gi), gi)
    sgn = torch.where(skip[:, :, None], torch.zeros_like(sgn), sgn)
    if qtype == "IQ2_XXS":
        w0 = gi[:, :, 0] | (gi[:, :, 1] << 8) | (gi[:, :, 2] << 16) | (gi[:, :, 3] << 24)
        w1 = sgn[:, :, 0] | (sgn[:, :, 1] << 7) | (sgn[:, :, 2] << 14) | (sgn[:, :, 3] << 21) | (lsc << 28)
        words = torch.stack([w0, w1], 2).reshape(nb, 16)  # q2[2 ib + 0], q2[2 ib + 1]
        body = torch.stack([(words >> s) & 0xFF for s in (0, 8, 16, 24)], 2).reshape(nb, 64)
        out[:, 2:] = torch.where(live[:, None], body, torch.zeros_like(body)).to(torch.uint8)
        return out.reshape(rows, cols // 256, nbytes)
    if qtype == "IQ2_XS":  # q2[2 ib + k] = grid index | signs << 9 (uint16, 32 per block), scales nibbles per ib pair
        words = (gi | (sgn << 9)).reshape(nb, 32)
        out[:, 2:66] = torch.stack([words & 0xFF, words >> 8], 2).reshape(nb, 64).to(torch.uint8)
        out[:, 66:74] = (lsc[:, 0::2] | (lsc[:, 1::2] << 4)).to(torch.uint8)
        out = torch.where(live[:, None], out, torch.zeros_like(out))  # a dead super-block is all zero (memset)
        return out.reshape(rows, cols // 256, nbytes)
    # IQ2_S: i8 = 2 ib + k: qs[i8] (low byte), signs at qs[32 + i8], qh[i8 / 4] 2 bits each, scales nibbles per ib
    g = gi.reshape(nb, 32)
    out[:, 2:34] = (g & 255).to(torch.uint8)
    out[:, 34:66] = sgn.reshape(nb, 32).to(torch.uint8)
    hi = (g >> 8).reshape(nb, 8, 4)
    out[:, 66:74] = (hi << (2 * torch.arange(4, device=x.device))).sum(-1).to(torch.uint8)
    pairs = lsc[:, 0::2] | (lsc[:, 1::2] << 4)
    out[:, 74:82] = torch.where(live[:, None], pairs, torch.zeros_like(pairs)).to(torch.uint8)
    out = torch.where(live[:, None], out, torch.zeros_like(out))  # a dead super-block is all zero (memset)
    return out.reshape(rows, cols // 256, nbytes)
