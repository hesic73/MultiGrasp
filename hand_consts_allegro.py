import numpy as np
import torch
import torch.nn.functional as F

from typing import List

"""
Palm: base_link
Thumb: link_12.0, link_13.0, link_14.0, link_15.0, link_15.0_tip
Index: link_0.0, link_1.0, link_2.0, link_3.0, link_3.0_tip
Middle: link_4.0, link_5.0, link_6.0, link_7.0, link_7.0_tip
Ring: link_8.0, link_9.0, link_10.0, link_11.0, link_11.0_tip

"""


"""
Contact areas:

Palm:
base_link: [4284,4262,4298,4292]
base_link: [4262,4248,4230,4246]
Thumb:
link_14.0: [12,2,15,14]
link_15.0: [37,54,52,39]
Index:
link_1.0: [6,15,14,8]
link_2.0: [320,432,430,319]
Middle:
link_5.0: [6,15,14,8]
link_6.0: [320,432,430,319]
Ring:
link_9.0: [6,15,14,8]
link_10.0: [320,432,430,319]
"""


def get_contact_pool(contact: List[int]) -> List[int]:
    """get indices of contact areas (defined in contact_*.json)

    Args:
        contact (List[int]): contact hand parts (0: palm, 1: thumb, 2: index, 3: middle, 4: ring)

    Returns:
        List[int]: indices of contact areas
    """
    idxs_pool = []

    if 0 in contact:  # Palm
        idxs_pool += [0, 1]
    if 1 in contact:  # Thumb
        idxs_pool += [2, 3]
    if 2 in contact:  # Index
        idxs_pool += [4, 5]
    if 3 in contact:  # Middle
        idxs_pool += [6, 7]
    if 4 in contact:  # Ring
        idxs_pool += [8, 9]

    return idxs_pool


contact_groups = [
    [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
]
