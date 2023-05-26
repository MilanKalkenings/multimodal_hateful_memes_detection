import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score

from model_ablation import confusion_indices, plot_confusion_matrix
from reproducibility import make_reproducible

make_reproducible(seed=1)

# find image confounders
'''
# train / test confounder analysis
train_path = "../hateful_memes_data/preprocessed_train_manual_split_v2.pkl"
test_path = "../hateful_memes_data/preprocessed_test_manual_split_v2.pkl"
df_test = pd.read_pickle(test_path)
df_train = pd.read_pickle(train_path)
hash_thresh = 7
all_correct, correct_0, correct_1, false_0, false_1 = model_ablation.confusion_indices(
    data_path=test_path,
    probas_path="../final_runs/hm/final/_token_embeddings_object_embeddings_kg_embeddings_vit_embeddings_text_sentiment_embeddings7.07/test_pred_probas_epoch20.txt")
model_ablation.plot_confusion_matrix(correct_0=correct_0, correct_1=correct_1, false_0=false_0, false_1=false_1, plot_path="../final_runs/hm/final/confusion_matrix.png")
test_images = df_test["image_path"]
train_images = df_train["image_path"]
train_image_hashes = train_images.apply(model_ablation.get_img_hash)
#train_image_sizes = train_images.apply(get_img_size)
print(train_image_hashes)
for i, image in enumerate(test_images):
    test_image = Image.open(image).convert("RGB")
    test_image_hash = imagehash.average_hash(test_image)
    test_image_size = test_image.size
    candidates = []
    print(i)
    for j, hash in enumerate(train_image_hashes):
        if hash - test_image_hash < hash_thresh:
            candidate = train_images.loc[j]
            candidates.append((j, df_train["label"][j], candidate))
            #if train_image_sizes[i] == test_image_size:
            #    candidate = train_images.loc[i]
            #    candidates.append(candidate)
    if len(candidates) > 0:
        print(candidates)
        print("test class", df_test["label"][i])
        plt.imshow(test_image)
        plt.show()

test_ids = [0, 2, 7, 8, 8, 8, 8, 8, 15, 17, 21, 26, 34, 38, 42, 42, 42, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47,
            47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 62, 73, 73, 73, 75, 78,
            79, 84, 85, 90, 92, 102, 103, 106, 106, 106, 108, 108, 108, 109, 112, 115, 122, 122, 122, 122, 122, 122,
            122, 122, 122, 122, 124, 124, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133,
            133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 141, 142, 142, 142,
            142, 142, 143, 143, 143, 145, 146, 150, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155,
            156, 158, 160, 164, 164, 164, 168, 170, 170, 170, 174, 174, 177, 177, 180, 180, 188, 189, 195, 200, 200,
            200, 201, 203, 203, 203, 203, 203, 204, 204, 204, 206, 208, 212, 212, 212, 213, 214, 216, 217, 217, 217,
            217, 222, 228, 232, 235, 236, 242, 243, 243, 243, 250, 250, 260, 263, 263, 267, 269, 269, 269, 271, 271,
            271, 271, 271, 271, 273, 274, 281, 283, 285, 285, 285, 285, 285, 285, 285, 285, 285, 285, 285, 285, 285,
            285, 285, 285, 285, 285, 287, 295, 295, 295, 295, 295, 295, 295, 295, 295, 295, 295, 295, 295, 295, 295,
            295, 295, 295, 295, 297, 298, 298, 299, 300, 303, 308, 308, 308, 308, 308, 308, 308, 308, 308, 308, 308,
            308, 308, 308, 308, 308, 308, 318, 320, 322, 326, 329, 330, 330, 330, 332, 336, 336, 336, 339, 340, 343,
            345, 349, 351, 351, 352, 354, 356, 356, 360, 360, 360, 360, 362, 362, 362, 362, 367, 368, 369, 369, 369,
            369, 369, 369, 369, 369, 369, 369, 369, 369, 369, 369, 369, 369, 375, 375, 375, 375, 386, 387, 393, 393,
            393, 393, 393, 393, 396, 411, 413, 414, 414, 414, 414, 414, 421, 423, 423, 423, 423, 423, 423, 423, 423,
            423, 423, 423, 423, 423, 423, 423, 423, 423, 423, 423, 424, 426, 428, 436, 436, 438, 439, 441, 446, 446,
            447, 450, 451, 452, 456, 457, 457, 457, 457, 458, 460, 460, 464, 470, 481, 483, 489, 490, 491, 493, 493,
            494, 494, 494, 498, 498, 498, 498, 501, 505, 506, 506, 509, 513, 517, 519, 521, 521, 521, 525, 526, 528,
            532, 533, 533, 533, 533, 533, 533, 533, 534, 537, 537, 537, 537, 541, 548, 548, 548, 548, 553, 553, 553,
            559, 559, 569, 576, 577, 577, 580, 584, 586, 587, 594, 595, 596, 597, 597, 597, 597, 599, 600, 600, 606,
            608, 609, 613, 615, 616, 616, 616, 616, 621, 626, 626, 627, 629, 629, 629, 629, 634, 634, 635, 636, 637,
            641, 644, 644, 644, 644, 644, 644, 644, 644, 644, 644, 644, 644, 644, 644, 644, 644, 644, 644, 644, 644,
            646, 647, 647, 647, 647, 647, 647, 648, 651, 652, 660, 665, 665, 667, 670, 673, 673, 673, 673, 673, 673,
            673, 673, 673, 673, 673, 673, 673, 673, 673, 673, 673, 673, 675, 675, 675, 675, 675, 675, 675, 675, 675,
            675, 675, 675, 675, 675, 675, 675, 675, 680, 681, 686, 690, 690, 694, 699, 700, 700, 700, 706, 709, 711,
            719, 720, 723, 725, 731, 731, 734, 752, 759, 759, 761, 763, 764, 764, 766, 768, 768, 770, 777, 780, 782,
            787, 791, 797, 797, 797, 797, 797, 797, 797, 797, 797, 797, 801, 803, 803, 803, 813, 821, 825, 825, 825,
            825, 825, 825, 825, 825, 827, 833, 833, 833, 833, 833, 833, 833, 833, 833, 833, 833, 833, 833, 833, 833,
            833, 833, 833, 846, 849, 850, 850, 850, 850, 852, 852, 852, 852, 852, 852, 852, 852, 852, 852, 852, 852,
            852, 852, 852, 852, 852, 852, 852, 856, 856, 856, 856, 856, 859, 861, 862, 862, 868, 868, 868, 868, 869,
            869, 870, 877, 882, 888, 888, 888, 890, 895, 896, 896, 898, 899]
train_ids = [2423, 6518, 4203, 927, 1015, 4366, 6495, 6940, 4629, 6219, 6717, 193, 6257, 5654, 1719, 5826, 6833, 125,
             606, 808, 1076, 1078, 1499, 1699, 1795, 2090, 2263, 2365, 2503, 2592, 2594, 2604, 2714, 2898, 3145, 3504,
             3807, 3821, 3836, 3873, 4028, 4842, 5733, 5812, 5837, 6396, 6744, 6760, 6993, 3075, 3495, 348, 865, 4377,
             4191, 1469, 4245, 1132, 952, 2498, 4562, 2040, 4802, 1813, 2335, 3566, 2328, 3115, 6264, 5545, 3453, 276,
             544, 589, 777, 1081, 3245, 3899, 4372, 6162, 6409, 6864, 3056, 6644, 125, 606, 808, 1076, 1078, 1499, 1699,
             1795, 2090, 2263, 2365, 2503, 2592, 2594, 2604, 2714, 2898, 3145, 3504, 3807, 3821, 3836, 3873, 4028, 4842,
             5733, 5812, 5837, 6396, 6744, 6760, 6993, 4822, 2957, 3268, 5528, 6056, 6774, 3917, 4040, 6836, 6186, 6105,
             2133, 36, 864, 1675, 1994, 2916, 3029, 3044, 3110, 3866, 4453, 5490, 6353, 6570, 5458, 3554, 2587, 3539,
             4156, 4552, 4729, 272, 3682, 5278, 69, 1333, 1001, 1560, 3056, 6644, 87, 2648, 7155, 1531, 3308, 5660,
             3134, 1420, 3515, 5049, 5228, 7087, 4657, 6119, 6336, 2881, 3646, 492, 721, 2951, 1187, 600, 6844, 3317,
             4717, 5323, 5327, 5249, 4653, 4145, 1342, 1498, 1052, 1598, 3095, 7194, 1871, 6939, 767, 857, 3183, 5771,
             279, 4693, 6821, 1130, 1434, 2840, 3247, 3797, 6096, 6395, 56, 985, 4007, 101, 169, 335, 917, 1787, 2528,
             3809, 4219, 4502, 4611, 4705, 5068, 5970, 6223, 6721, 6906, 7002, 7152, 3264, 82, 773, 1936, 2783, 3023,
             3071, 3336, 3955, 4039, 4698, 4829, 4869, 5353, 5642, 5664, 5862, 6293, 6411, 6619, 5245, 2319, 4692, 3420,
             5415, 4465, 62, 487, 650, 1196, 1226, 1822, 2202, 2317, 2462, 3350, 3665, 4141, 4374, 4572, 5301, 6905,
             7100, 24, 998, 2069, 1077, 3653, 834, 834, 3794, 3413, 2774, 4541, 6611, 1002, 2069, 3554, 3960, 2860, 339,
             5588, 2842, 1220, 2634, 3013, 484, 726, 1299, 4534, 4688, 699, 3640, 4230, 3146, 4711, 152, 940, 1005,
             1368, 1843, 2511, 2922, 2956, 3609, 4784, 5051, 5445, 6099, 6288, 6559, 6700, 673, 4424, 4551, 4745, 568,
             6818, 589, 3245, 3899, 6162, 6409, 6864, 1018, 3345, 945, 927, 1015, 4366, 6495, 6940, 4755, 82, 773, 1936,
             2783, 3023, 3071, 3336, 3955, 4039, 4698, 4829, 4869, 5353, 5642, 5664, 5862, 6293, 6411, 6619, 3398, 6017,
             6186, 2812, 4859, 5233, 2119, 3550, 216, 5386, 4933, 6797, 3231, 2998, 4178, 2824, 3964, 4187, 4974, 7105,
             1935, 6397, 7077, 1552, 2707, 4171, 4794, 4002, 2713, 1871, 6939, 2863, 4115, 4775, 2048, 5554, 6385, 6564,
             551, 835, 468, 6397, 1583, 843, 3569, 3075, 1616, 2629, 5372, 996, 3064, 2910, 2557, 796, 1817, 3369, 3781,
             4303, 4934, 5955, 3832, 673, 4424, 4551, 4745, 7001, 2048, 5554, 6385, 6564, 2157, 4490, 6457, 312, 3717,
             6493, 3458, 1853, 3465, 1406, 1421, 429, 307, 1190, 4383, 6346, 4027, 4907, 6326, 6533, 4016, 182, 5333,
             6480, 2576, 5396, 583, 6638, 2363, 2864, 5090, 5425, 3550, 769, 758, 2157, 1102, 2253, 2756, 6075, 182,
             5333, 6891, 6116, 631, 2825, 310, 437, 636, 1625, 1942, 3093, 3107, 3306, 3792, 4080, 4164, 4410, 4584,
             5008, 5399, 5422, 5841, 6520, 7038, 7060, 4574, 581, 2354, 2520, 2856, 6391, 6435, 4929, 4721, 2461, 3150,
             1658, 2316, 2452, 6148, 101, 169, 335, 917, 1787, 2528, 3809, 4219, 4502, 4611, 4705, 5068, 5970, 6223,
             6721, 6906, 7002, 7152, 62, 487, 650, 1196, 1226, 1822, 2202, 2317, 2462, 3350, 3665, 4141, 4374, 4572,
             5301, 6905, 7100, 2187, 6726, 6082, 687, 4538, 4571, 5776, 2381, 4942, 6018, 5224, 1481, 1680, 1709, 4600,
             3605, 2944, 537, 5009, 5433, 5348, 2077, 6823, 267, 2979, 1470, 4299, 2179, 5901, 6687, 1689, 4544, 5175,
             5474, 2148, 3438, 544, 589, 777, 1081, 3245, 3899, 4372, 6162, 6409, 6864, 753, 2328, 3115, 6264, 6247,
             2023, 567, 1869, 2761, 2875, 3628, 4812, 5011, 6459, 6469, 101, 169, 335, 917, 1787, 2528, 3809, 4219,
             4502, 4611, 4705, 5068, 5970, 6223, 6721, 6906, 7002, 7152, 4911, 981, 2358, 4391, 4616, 5072, 82, 773,
             1936, 2783, 3023, 3071, 3336, 3955, 4039, 4698, 4829, 4869, 5353, 5642, 5664, 5862, 6293, 6411, 6619, 285,
             879, 3459, 5000, 6972, 712, 2216, 4308, 5750, 1102, 2253, 2756, 6075, 4722, 6597, 3958, 1044, 4347, 2142,
             3210, 4461, 1778, 3686, 2009, 2593, 1551, 3563]
test_classes = [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1,
              0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
              0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
              0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
              1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
              0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
              1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
train_classes = [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
               0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1,
               0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
               0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1,
               1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1,
               1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
               1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1,
               0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0,
               1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1,
               1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1,
               1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0,
               0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
               1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0,
               1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1,
               0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1,
               0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1,
               1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
               1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]
train_ids = pd.Series(train_ids, name="train_id")
test_ids = pd.Series(test_ids, name="test_id")
train_class = pd.Series(train_classes, name="train_class")
test_class = pd.Series(test_classes, name="test_class")
df = pd.concat([train_ids, test_ids, train_class, test_class], axis=1)
df.to_csv("../hateful_memes_data/train_test_image_confounders.csv", index=False)
'''

# find text confounders
'''
test_ids = []
train_ids = []
test_classes = []
train_classes = []
for i, test_text in enumerate(df_test["text"]):
    print(i)
    for j, train_text in enumerate(df_train["text"]):
        if test_text == train_text:
            test_ids.append(i)
            train_ids.append(j)
            test_classes.append(df_test["label"][i])
            train_classes.append(df_train["label"][j])
            print(i, j)
train_ids = pd.Series(train_ids, name="train_id")
test_ids = pd.Series(test_ids, name="test_id")
train_class = pd.Series(train_classes, name="train_class")
test_class = pd.Series(test_classes, name="test_class")
df = pd.concat([train_ids, test_ids, train_class, test_class], axis=1)
print(df)
df.to_csv("../hateful_memes_data/train_test_text_confounders.csv", index=False)
'''

# HM FIRST
test_path = "../hateful_memes_data/final_preprocessed_test.pkl"
# parameters_path = "../final_runs/standard_parameters.pkl"
model_path = "../final_runs/hm/hm_early_fusion test 7146/seed1/1_token_sentiment_kg_vit/model_epoch_12.pkl"
test_probas_path_seed1 = "../final_runs/hm/hm_early_fusion test 7146/seed1/1_token_sentiment_kg_vit/test_pred_probas_epoch12.txt"
test_probas_path_seed2 = "../final_runs/hm/hm_early_fusion test 7146/seed2/2_token_sentiment_kg_vit/test_pred_probas_epoch12.txt"
test_probas_path_seed3 = "../final_runs/hm/hm_early_fusion test 7146/seed3/3_token_sentiment_kg_vit/test_pred_probas_epoch12.txt"
test_df = pd.read_pickle(test_path)
print(len(test_df))

# load data
train_test_image_confounders = pd.read_csv("../hateful_memes_data/train_test_image_confounders.csv")
train_test_image_confounders["image_confounder"] = 1
train_test_text_confounders = pd.read_csv("../hateful_memes_data/train_test_text_confounders.csv")
train_test_text_confounders["image_confounder"] = 0
train_test_confounders = pd.concat([train_test_image_confounders, train_test_text_confounders])
original_test_labels = test_df["label"]
original_test_probas_1 = np.loadtxt(test_probas_path_seed1)
original_test_probas_2 = np.loadtxt(test_probas_path_seed2)
original_test_probas_3 = np.loadtxt(test_probas_path_seed3)

########################################################################################################################
# text confounders

benign_confounder_indices = set(train_test_text_confounders.loc[
                                train_test_text_confounders["train_class"] != train_test_text_confounders["test_class"],
                                :]["test_id"])
text_confounder_indices = benign_confounder_indices
print("text confounders:", len(benign_confounder_indices))

# seed1
test_labels_1 = []
test_probas_1 = []
for i in range(len(original_test_labels)):
    if i in benign_confounder_indices:
        test_labels_1.append(original_test_labels[i])
        test_probas_1.append(original_test_probas_1[i])
all_correct_1, correct_0_1, correct_1_1, false_0_1, false_1_1 = confusion_indices(ground_truth=np.array(test_labels_1),
                                                                                  probas=np.array(test_probas_1))
roc_auc_1 = roc_auc_score(y_true=test_labels_1, y_score=test_probas_1)
recall_score_1 = recall_score(y_true=test_labels_1, y_pred=np.round(test_probas_1))
precision_score_1 = precision_score(y_true=test_labels_1, y_pred=np.round(test_probas_1))

# seed2
test_labels_2 = []
test_probas_2 = []
for i in range(len(original_test_labels)):
    if i in benign_confounder_indices:
        test_labels_2.append(original_test_labels[i])
        test_probas_2.append(original_test_probas_2[i])
all_correct_2, correct_0_2, correct_1_2, false_0_2, false_1_2 = confusion_indices(ground_truth=np.array(test_labels_2),
                                                                                  probas=np.array(test_probas_2))
roc_auc_2 = roc_auc_score(y_true=test_labels_2, y_score=test_probas_2)
recall_score_2 = recall_score(y_true=test_labels_2, y_pred=np.round(test_probas_2))
precision_score_2 = precision_score(y_true=test_labels_2, y_pred=np.round(test_probas_2))

# seed3
test_labels_3 = []
test_probas_3 = []
for i in range(len(original_test_labels)):
    if i in benign_confounder_indices:
        test_labels_3.append(original_test_labels[i])
        test_probas_3.append(original_test_probas_3[i])
all_correct_3, correct_0_3, correct_1_3, false_0_3, false_1_3 = confusion_indices(ground_truth=np.array(test_labels_3),
                                                                                  probas=np.array(test_probas_3))
roc_auc_3 = roc_auc_score(y_true=test_labels_3, y_score=test_probas_3)
recall_score_3 = recall_score(y_true=test_labels_3, y_pred=np.round(test_probas_3))
precision_score_3 = precision_score(y_true=test_labels_3, y_pred=np.round(test_probas_3))

correct_0 = int(np.round((len(correct_0_1) + len(correct_0_2) + len(correct_0_3)) / 3))
correct_1 = int(np.round((len(correct_1_1) + len(correct_1_2) + len(correct_1_3)) / 3))
false_0 = int(np.round((len(false_0_1) + len(false_0_2) + len(false_0_3)) / 3))
false_1 = int(np.round((len(false_1_1) + len(false_1_2) + len(false_1_3)) / 3))
roc_auc = (roc_auc_1 + roc_auc_2 + roc_auc_3) / 3
recall = (recall_score_1 + recall_score_2 + recall_score_3) / 3
precision = (precision_score_1 + precision_score_2 + precision_score_3) / 3
print("roc_auc:", roc_auc, "recall:", recall, "precision:", precision)

plot_confusion_matrix(correct_0=correct_0,
                      correct_1=correct_1,
                      false_0=false_0,
                      false_1=false_1,
                      plot_path="../ausarbeitung/figures/benign_confounders/text_confounders.png",
                      roc_auc_score=roc_auc)

########################################################################################################################
# image confounders
benign_confounder_indices = set(train_test_image_confounders.loc[
                                train_test_image_confounders["train_class"] != train_test_image_confounders[
                                    "test_class"], :]["test_id"])
image_confounder_indices = benign_confounder_indices
print("image confounders:", len(benign_confounder_indices))

# seed1
test_labels_1 = []
test_probas_1 = []
for i in range(len(original_test_labels)):
    if i in benign_confounder_indices:
        test_labels_1.append(original_test_labels[i])
        test_probas_1.append(original_test_probas_1[i])
all_correct_1, correct_0_1, correct_1_1, false_0_1, false_1_1 = confusion_indices(ground_truth=np.array(test_labels_1),
                                                                                  probas=np.array(test_probas_1))
roc_auc_1 = roc_auc_score(y_true=test_labels_1, y_score=test_probas_1)
recall_score_1 = recall_score(y_true=test_labels_1, y_pred=np.round(test_probas_1))
precision_score_1 = precision_score(y_true=test_labels_1, y_pred=np.round(test_probas_1))

# seed2
test_labels_2 = []
test_probas_2 = []
for i in range(len(original_test_labels)):
    if i in benign_confounder_indices:
        test_labels_2.append(original_test_labels[i])
        test_probas_2.append(original_test_probas_2[i])
all_correct_2, correct_0_2, correct_1_2, false_0_2, false_1_2 = confusion_indices(ground_truth=np.array(test_labels_2),
                                                                                  probas=np.array(test_probas_2))
roc_auc_2 = roc_auc_score(y_true=test_labels_2, y_score=test_probas_2)
recall_score_2 = recall_score(y_true=test_labels_2, y_pred=np.round(test_probas_2))
precision_score_2 = precision_score(y_true=test_labels_2, y_pred=np.round(test_probas_2))

# seed3
test_labels_3 = []
test_probas_3 = []
for i in range(len(original_test_labels)):
    if i in benign_confounder_indices:
        test_labels_3.append(original_test_labels[i])
        test_probas_3.append(original_test_probas_3[i])
all_correct_3, correct_0_3, correct_1_3, false_0_3, false_1_3 = confusion_indices(ground_truth=np.array(test_labels_3),
                                                                                  probas=np.array(test_probas_3))
roc_auc_3 = roc_auc_score(y_true=test_labels_3, y_score=test_probas_3)
recall_score_3 = recall_score(y_true=test_labels_3, y_pred=np.round(test_probas_3))
precision_score_3 = precision_score(y_true=test_labels_3, y_pred=np.round(test_probas_3))

correct_0 = int(np.round((len(correct_0_1) + len(correct_0_2) + len(correct_0_3)) / 3))
correct_1 = int(np.round((len(correct_1_1) + len(correct_1_2) + len(correct_1_3)) / 3))
false_0 = int(np.round((len(false_0_1) + len(false_0_2) + len(false_0_3)) / 3))
false_1 = int(np.round((len(false_1_1) + len(false_1_2) + len(false_1_3)) / 3))
roc_auc = (roc_auc_1 + roc_auc_2 + roc_auc_3) / 3
recall = (recall_score_1 + recall_score_2 + recall_score_3) / 3
precision = (precision_score_1 + precision_score_2 + precision_score_3) / 3
print("roc_auc:", roc_auc, "recall:", recall, "precision:", precision)

plot_confusion_matrix(correct_0=correct_0,
                      correct_1=correct_1,
                      false_0=false_0,
                      false_1=false_1,
                      plot_path="../ausarbeitung/figures/benign_confounders/image_confounders.png",
                      roc_auc_score=roc_auc)

########################################################################################################################
# all confounders
benign_confounder_indices = text_confounder_indices.union(image_confounder_indices)
all_confounders = benign_confounder_indices

print("all confounders:", len(benign_confounder_indices))

# seed1
test_labels_1 = []
test_probas_1 = []
for i in range(len(original_test_labels)):
    if i in benign_confounder_indices:
        test_labels_1.append(original_test_labels[i])
        test_probas_1.append(original_test_probas_1[i])
all_correct_1, correct_0_1, correct_1_1, false_0_1, false_1_1 = confusion_indices(ground_truth=np.array(test_labels_1),
                                                                                  probas=np.array(test_probas_1))
roc_auc_1 = roc_auc_score(y_true=test_labels_1, y_score=test_probas_1)
recall_score_1 = recall_score(y_true=test_labels_1, y_pred=np.round(test_probas_1))
precision_score_1 = precision_score(y_true=test_labels_1, y_pred=np.round(test_probas_1))

# seed2
test_labels_2 = []
test_probas_2 = []
for i in range(len(original_test_labels)):
    if i in benign_confounder_indices:
        test_labels_2.append(original_test_labels[i])
        test_probas_2.append(original_test_probas_2[i])
all_correct_2, correct_0_2, correct_1_2, false_0_2, false_1_2 = confusion_indices(ground_truth=np.array(test_labels_2),
                                                                                  probas=np.array(test_probas_2))
roc_auc_2 = roc_auc_score(y_true=test_labels_2, y_score=test_probas_2)
recall_score_2 = recall_score(y_true=test_labels_2, y_pred=np.round(test_probas_2))
precision_score_2 = precision_score(y_true=test_labels_2, y_pred=np.round(test_probas_2))

# seed3
test_labels_3 = []
test_probas_3 = []
for i in range(len(original_test_labels)):
    if i in benign_confounder_indices:
        test_labels_3.append(original_test_labels[i])
        test_probas_3.append(original_test_probas_3[i])
all_correct_3, correct_0_3, correct_1_3, false_0_3, false_1_3 = confusion_indices(ground_truth=np.array(test_labels_3),
                                                                                  probas=np.array(test_probas_3))
roc_auc_3 = roc_auc_score(y_true=test_labels_3, y_score=test_probas_3)
recall_score_3 = recall_score(y_true=test_labels_3, y_pred=np.round(test_probas_3))
precision_score_3 = precision_score(y_true=test_labels_3, y_pred=np.round(test_probas_3))

correct_0 = int(np.round((len(correct_0_1) + len(correct_0_2) + len(correct_0_3)) / 3))
correct_1 = int(np.round((len(correct_1_1) + len(correct_1_2) + len(correct_1_3)) / 3))
false_0 = int(np.round((len(false_0_1) + len(false_0_2) + len(false_0_3)) / 3))
false_1 = int(np.round((len(false_1_1) + len(false_1_2) + len(false_1_3)) / 3))
roc_auc = (roc_auc_1 + roc_auc_2 + roc_auc_3) / 3
recall = (recall_score_1 + recall_score_2 + recall_score_3) / 3
precision = (precision_score_1 + precision_score_2 + precision_score_3) / 3
print("roc_auc:", roc_auc, "recall:", recall, "precision:", precision)

plot_confusion_matrix(correct_0=correct_0,
                      correct_1=correct_1,
                      false_0=false_0,
                      false_1=false_1,
                      plot_path="../ausarbeitung/figures/benign_confounders/all_confounders.png",
                      roc_auc_score=roc_auc)

########################################################################################################################
# all non-confounders
benign_confounder_indices = set(range(900)) - all_confounders

print("all non-confounders:", len(benign_confounder_indices))

# seed1
test_labels_1 = []
test_probas_1 = []
for i in range(len(original_test_labels)):
    if i in benign_confounder_indices:
        test_labels_1.append(original_test_labels[i])
        test_probas_1.append(original_test_probas_1[i])
all_correct_1, correct_0_1, correct_1_1, false_0_1, false_1_1 = confusion_indices(ground_truth=np.array(test_labels_1),
                                                                                  probas=np.array(test_probas_1))
roc_auc_1 = roc_auc_score(y_true=test_labels_1, y_score=test_probas_1)
recall_score_1 = recall_score(y_true=test_labels_1, y_pred=np.round(test_probas_1))
precision_score_1 = precision_score(y_true=test_labels_1, y_pred=np.round(test_probas_1))

# seed2
test_labels_2 = []
test_probas_2 = []
for i in range(len(original_test_labels)):
    if i in benign_confounder_indices:
        test_labels_2.append(original_test_labels[i])
        test_probas_2.append(original_test_probas_2[i])
all_correct_2, correct_0_2, correct_1_2, false_0_2, false_1_2 = confusion_indices(ground_truth=np.array(test_labels_2),
                                                                                  probas=np.array(test_probas_2))
roc_auc_2 = roc_auc_score(y_true=test_labels_2, y_score=test_probas_2)
recall_score_2 = recall_score(y_true=test_labels_2, y_pred=np.round(test_probas_2))
precision_score_2 = precision_score(y_true=test_labels_2, y_pred=np.round(test_probas_2))

# seed3
test_labels_3 = []
test_probas_3 = []
for i in range(len(original_test_labels)):
    if i in benign_confounder_indices:
        test_labels_3.append(original_test_labels[i])
        test_probas_3.append(original_test_probas_3[i])
all_correct_3, correct_0_3, correct_1_3, false_0_3, false_1_3 = confusion_indices(ground_truth=np.array(test_labels_3),
                                                                                  probas=np.array(test_probas_3))
roc_auc_3 = roc_auc_score(y_true=test_labels_3, y_score=test_probas_3)
recall_score_3 = recall_score(y_true=test_labels_3, y_pred=np.round(test_probas_3))
precision_score_3 = precision_score(y_true=test_labels_3, y_pred=np.round(test_probas_3))

correct_0 = int(np.round((len(correct_0_1) + len(correct_0_2) + len(correct_0_3)) / 3))
correct_1 = int(np.round((len(correct_1_1) + len(correct_1_2) + len(correct_1_3)) / 3))
false_0 = int(np.round((len(false_0_1) + len(false_0_2) + len(false_0_3)) / 3))
false_1 = int(np.round((len(false_1_1) + len(false_1_2) + len(false_1_3)) / 3))
roc_auc = (roc_auc_1 + roc_auc_2 + roc_auc_3) / 3
recall = (recall_score_1 + recall_score_2 + recall_score_3) / 3
precision = (precision_score_1 + precision_score_2 + precision_score_3) / 3
print("roc_auc:", roc_auc, "recall:", recall, "precision:", precision)

plot_confusion_matrix(correct_0=correct_0,
                      correct_1=correct_1,
                      false_0=false_0,
                      false_1=false_1,
                      plot_path="../ausarbeitung/figures/benign_confounders/all_non-confounders.png",
                      roc_auc_score=roc_auc)

########################################################################################################################
# all
benign_confounder_indices = range(900)

print("all:", len(benign_confounder_indices))

# seed1
test_labels_1 = []
test_probas_1 = []
for i in range(len(original_test_labels)):
    if i in benign_confounder_indices:
        test_labels_1.append(original_test_labels[i])
        test_probas_1.append(original_test_probas_1[i])
all_correct_1, correct_0_1, correct_1_1, false_0_1, false_1_1 = confusion_indices(ground_truth=np.array(test_labels_1),
                                                                                  probas=np.array(test_probas_1))
roc_auc_1 = roc_auc_score(y_true=test_labels_1, y_score=test_probas_1)
recall_score_1 = recall_score(y_true=test_labels_1, y_pred=np.round(test_probas_1))
precision_score_1 = precision_score(y_true=test_labels_1, y_pred=np.round(test_probas_1))

# seed2
test_labels_2 = []
test_probas_2 = []
for i in range(len(original_test_labels)):
    if i in benign_confounder_indices:
        test_labels_2.append(original_test_labels[i])
        test_probas_2.append(original_test_probas_2[i])
all_correct_2, correct_0_2, correct_1_2, false_0_2, false_1_2 = confusion_indices(ground_truth=np.array(test_labels_2),
                                                                                  probas=np.array(test_probas_2))
roc_auc_2 = roc_auc_score(y_true=test_labels_2, y_score=test_probas_2)
recall_score_2 = recall_score(y_true=test_labels_2, y_pred=np.round(test_probas_2))
precision_score_2 = precision_score(y_true=test_labels_2, y_pred=np.round(test_probas_2))

# seed3
test_labels_3 = []
test_probas_3 = []
for i in range(len(original_test_labels)):
    if i in benign_confounder_indices:
        test_labels_3.append(original_test_labels[i])
        test_probas_3.append(original_test_probas_3[i])
all_correct_3, correct_0_3, correct_1_3, false_0_3, false_1_3 = confusion_indices(ground_truth=np.array(test_labels_3),
                                                                                  probas=np.array(test_probas_3))
roc_auc_3 = roc_auc_score(y_true=test_labels_3, y_score=test_probas_3)
recall_score_3 = recall_score(y_true=test_labels_3, y_pred=np.round(test_probas_3))
precision_score_3 = precision_score(y_true=test_labels_3, y_pred=np.round(test_probas_3))

correct_0 = int(np.round((len(correct_0_1) + len(correct_0_2) + len(correct_0_3)) / 3))
correct_1 = int(np.round((len(correct_1_1) + len(correct_1_2) + len(correct_1_3)) / 3))
false_0 = int(np.round((len(false_0_1) + len(false_0_2) + len(false_0_3)) / 3))
false_1 = int(np.round((len(false_1_1) + len(false_1_2) + len(false_1_3)) / 3))
roc_auc = (roc_auc_1 + roc_auc_2 + roc_auc_3) / 3
recall = (recall_score_1 + recall_score_2 + recall_score_3) / 3
precision = (precision_score_1 + precision_score_2 + precision_score_3) / 3
print("roc_auc:", roc_auc, "recall:", recall, "precision:", precision)

plot_confusion_matrix(correct_0=correct_0,
                      correct_1=correct_1,
                      false_0=false_0,
                      false_1=false_1,
                      plot_path="../ausarbeitung/figures/benign_confounders/hm.png",
                      roc_auc_score=roc_auc)

########################################################################################################################
test_path = "../misogyny_data/final_preprocessed_test.csv"

test_probas_path_seed1 = "../final_runs/mis/mis_early_fusion test_8421/seed1/1_token_roi_kg/test_pred_probas_epoch2.txt"
test_probas_path_seed2 = "../final_runs/mis/mis_early_fusion test_8421/seed2/2_token_roi_kg/test_pred_probas_epoch2.txt"
test_probas_path_seed3 = "../final_runs/mis/mis_early_fusion test_8421/seed3/3_token_roi_kg/test_pred_probas_epoch2.txt"

# MIS
# all
test_df = pd.read_pickle(test_path)
test_labels = test_df["label"]

print("mis:", len(test_labels))

# seed1
test_probas_1 = np.loadtxt(test_probas_path_seed1)
all_correct_1, correct_0_1, correct_1_1, false_0_1, false_1_1 = confusion_indices(ground_truth=np.array(test_labels),
                                                                                  probas=np.array(test_probas_1))
roc_auc_1 = roc_auc_score(y_true=test_labels, y_score=test_probas_1)
recall_score_1 = recall_score(y_true=test_labels, y_pred=np.round(test_probas_1))
precision_score_1 = precision_score(y_true=test_labels, y_pred=np.round(test_probas_1))

# seed2
test_probas_2 = np.loadtxt(test_probas_path_seed2)
all_correct_2, correct_0_2, correct_1_2, false_0_2, false_1_2 = confusion_indices(ground_truth=np.array(test_labels),
                                                                                  probas=np.array(test_probas_2))
roc_auc_2 = roc_auc_score(y_true=test_labels, y_score=test_probas_2)
recall_score_2 = recall_score(y_true=test_labels, y_pred=np.round(test_probas_2))
precision_score_2 = precision_score(y_true=test_labels, y_pred=np.round(test_probas_2))

# seed3
test_probas_3 = np.loadtxt(test_probas_path_seed3)
all_correct_3, correct_0_3, correct_1_3, false_0_3, false_1_3 = confusion_indices(ground_truth=np.array(test_labels),
                                                                                  probas=np.array(test_probas_3))
roc_auc_3 = roc_auc_score(y_true=test_labels, y_score=test_probas_3)
recall_score_3 = recall_score(y_true=test_labels, y_pred=np.round(test_probas_3))
precision_score_3 = precision_score(y_true=test_labels, y_pred=np.round(test_probas_3))

# seed3
test_probas_1 = np.loadtxt(test_probas_path_seed1)
all_correct_1, correct_0_1, correct_1_1, false_0_1, false_1_1 = confusion_indices(ground_truth=np.array(test_labels),
                                                                                  probas=np.array(test_probas_1))
roc_auc_1 = roc_auc_score(y_true=test_labels, y_score=test_probas_1)

correct_0 = int(np.round((len(correct_0_1) + len(correct_0_2) + len(correct_0_3)) / 3))
correct_1 = int(np.round((len(correct_1_1) + len(correct_1_2) + len(correct_1_3)) / 3))
false_0 = int(np.round((len(false_0_1) + len(false_0_2) + len(false_0_3)) / 3))
false_1 = int(np.round((len(false_1_1) + len(false_1_2) + len(false_1_3)) / 3))
roc_auc = (roc_auc_1 + roc_auc_2 + roc_auc_3) / 3
recall = (recall_score_1 + recall_score_2 + recall_score_3) / 3
precision = (precision_score_1 + precision_score_2 + precision_score_3) / 3
print("roc_auc:", roc_auc, "recall:", recall, "precision:", precision)

plot_confusion_matrix(correct_0=correct_0,
                      correct_1=correct_1,
                      false_0=false_0,
                      false_1=false_1,
                      plot_path="../ausarbeitung/figures/benign_confounders/mis.png",
                      roc_auc_score=roc_auc)
