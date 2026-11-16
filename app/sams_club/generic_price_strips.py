"""Dimensions and print centers from the KK240100-13 Generic strip drawing."""
from app.sams_club.holiday_price_strips import SamsHolidaySideGeometry, holiday_geometry_for_size


GENERIC_SIZES = {
    'generic_11': (11, (99, 396, 693)),
    'generic_15_25': (15.25, (108, 405, 702, 999)),
    'generic_19_25': (19.25, (99, 396, 693, 990, 1287)),
    'generic_23_5': (23.5, (108, 405, 702, 999, 1296, 1593)),
}


def geometry_for_size(size):
    if size not in GENERIC_SIZES:
        return holiday_geometry_for_size(size)
    width, centers = GENERIC_SIZES[size]
    return SamsHolidaySideGeometry(size, width, 1.9375, len(centers), 0, (1, 2, 3, 4))


def generic_boundaries(size):
    width, centers = GENERIC_SIZES[size]
    return (0, *( (a + b) / 2 for a, b in zip(centers, centers[1:])), width * 72)
