"""Dimensions and print centers from the KK240100-13 Generic strip drawing."""
from app.sams_club.holiday_price_strips import SamsHolidaySideGeometry, holiday_geometry_for_size


GENERIC_SIZES = {
    'generic_11': (11, (99, 396, 693)),
    'generic_15_25': (15.25, (108, 405, 702, 999)),
    'generic_19_25': (19.25, (99, 396, 693, 990, 1287)),
    'generic_23_5': (23.5, (108, 405, 702, 999, 1296, 1593)),
    'generic_kk250133_short': (
        30.6875,
        (89.816, 379.797, 669.778, 959.759,
         1249.740, 1539.721, 1829.702, 2119.682),
    ),
    'generic_kk250133_long': (
        38.9375,
        (96.946, 386.926, 676.906, 966.888, 1256.868, 1546.849,
         1836.830, 2126.812, 2416.792, 2706.773),
    ),
    'generic_two_card': (17, (306, 918)),
}


def geometry_for_size(size):
    if size not in GENERIC_SIZES:
        return holiday_geometry_for_size(size)
    width, centers = GENERIC_SIZES[size]
    height = 4 if size == 'generic_two_card' else 1.9375
    return SamsHolidaySideGeometry(size, width, height, len(centers), 0, (1, 2, 3, 4))


def generic_boundaries(size):
    width, centers = GENERIC_SIZES[size]
    return (0, *( (a + b) / 2 for a, b in zip(centers, centers[1:])), width * 72)
